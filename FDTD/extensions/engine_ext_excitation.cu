
#include "engine_ext_excitation.h"
#include "operator_ext_excitation.h"
#include "FDTD/engine_cuda_mgpu.h"

#include <cuda_runtime.h>
#include <cstdlib>

#include "hemi/hemi.h"
#include "hemi/launch.h"

#if 1

__global__
void kernelApply2VA(volatile FDTD_FLOAT *d_va, const exitation_point *ep, const FDTD_FLOAT* sig_va, const dim3 dim, const int* d_numTS, const int signalPeriod, const int N)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n >= N) return;

    ep = ep + n;

    // numTS comes from the device counter so this kernel is invariant across
    // graph replays; p is derived on-device (matches the host formula exactly:
    // a non-periodic excitation uses p=numTS+1, which makes the modulo a no-op).
    int numTS = *d_numTS;
    int p = signalPeriod;
    if (p <= 0) p = numTS + 1;

	int exc_pos = numTS - ep->delay;
	exc_pos *= (exc_pos>0);
	exc_pos %= p;
	exc_pos *= (exc_pos<(int)ep->length);
	int ny = ep->dir;

    int pos = ep->x * dim.y * dim.z + ep->y * dim.z + ep->z;

    // Atomic so that two excitation points landing on the same (cell,component)
    // — overlapping or soft sources sharing an edge — accumulate instead of
    // racing (the same lost-update hazard the TFSF port hit at box edges).
    atomicAdd((FDTD_FLOAT*)(d_va + 3 * pos + ny), ep->amp * sig_va[exc_pos]);
}


// ---- multi-GPU: per-slab launches (points already partitioned/translated) ----
void Engine_Ext_Excitation::Apply2VoltagesMg(Engine_cuda_mgpu* mg)
{
    int signalPeriod = 0;
    if (m_Op_Exc->m_Exc->GetSignalPeriod() > 0)
        signalPeriod = int(m_Op_Exc->m_Exc->GetSignalPeriod() / m_Op_Exc->m_Exc->GetTimestep());

    for (int g = 0; g < mg->NumSlabs(); ++g)
    {
        int n = mg_n_v[g];
        if (n <= 0) continue;
        const CudaSlabCtx &c = mg->Slab(g);
        cudaSetDevice(c.device);
        int threads = std::min(1024, n);
        int blocks = (n + threads - 1) / threads;
        kernelApply2VA<<<blocks, threads, 0, c.stream>>>(
            c.d_volt, mg_ep_v[g], mg_sig_v[g], c.local_dim, c.d_numTS, signalPeriod, n);
    }
}

void Engine_Ext_Excitation::Apply2CurrentMg(Engine_cuda_mgpu* mg)
{
    int signalPeriod = 0;
    if (m_Op_Exc->m_Exc->GetSignalPeriod() > 0)
        signalPeriod = int(m_Op_Exc->m_Exc->GetSignalPeriod() / m_Op_Exc->m_Exc->GetTimestep());

    for (int g = 0; g < mg->NumSlabs(); ++g)
    {
        int n = mg_n_a[g];
        if (n <= 0) continue;
        const CudaSlabCtx &c = mg->Slab(g);
        cudaSetDevice(c.device);
        int threads = std::min(1024, n);
        int blocks = (n + threads - 1) / threads;
        kernelApply2VA<<<blocks, threads, 0, c.stream>>>(
            c.d_curr, mg_ep_a[g], mg_sig_a[g], c.local_dim, c.d_numTS, signalPeriod, n);
    }
}

void Engine_Ext_Excitation::SetEngineMg(Engine_cuda_mgpu* mg)
{
    int nslab = mg->NumSlabs();
    mg_ep_v.assign(nslab, NULL); mg_ep_a.assign(nslab, NULL);
    mg_n_v.assign(nslab, 0);     mg_n_a.assign(nslab, 0);
    mg_sig_v.assign(nslab, NULL); mg_sig_a.assign(nslab, NULL);

    unsigned int length = m_Op_Exc->m_Exc->GetLength();
    FDTD_FLOAT* exc_volt = m_Op_Exc->m_Exc->GetVoltageSignal();
    FDTD_FLOAT* exc_curr = m_Op_Exc->m_Exc->GetCurrentSignal();

    for (int g = 0; g < nslab; ++g)
    {
        const CudaSlabCtx &c = mg->Slab(g);
        std::vector<exitation_point> pv, pa;

        for (unsigned int i = 0; i < m_Op_Exc->Volt_Count; ++i)
        {
            int x = (int)m_Op_Exc->Volt_index[0][i];
            if (x < c.x_start || x >= c.x_end) continue;
            exitation_point ep;
            ep.x = x - c.x_start + 1;   // slab-local incl. ghost offset
            ep.y = (int)m_Op_Exc->Volt_index[1][i];
            ep.z = (int)m_Op_Exc->Volt_index[2][i];
            ep.dir = m_Op_Exc->Volt_dir[i];
            ep.delay = (int)m_Op_Exc->Volt_delay[i];
            ep.length = (int)length;
            ep.amp = m_Op_Exc->Volt_amp[i];
            pv.push_back(ep);
        }
        for (unsigned int i = 0; i < m_Op_Exc->Curr_Count; ++i)
        {
            int x = (int)m_Op_Exc->Curr_index[0][i];
            if (x < c.x_start || x >= c.x_end) continue;
            exitation_point ep;
            ep.x = x - c.x_start + 1;
            ep.y = (int)m_Op_Exc->Curr_index[1][i];
            ep.z = (int)m_Op_Exc->Curr_index[2][i];
            ep.dir = m_Op_Exc->Curr_dir[i];
            ep.delay = (int)m_Op_Exc->Curr_delay[i];
            ep.length = (int)length;
            ep.amp = m_Op_Exc->Curr_amp[i];
            pa.push_back(ep);
        }

        cudaSetDevice(c.device);
        mg_n_v[g] = (int)pv.size();
        mg_n_a[g] = (int)pa.size();
        if (!pv.empty())
        {
            checkCuda(cudaMalloc(&mg_ep_v[g], pv.size() * sizeof(exitation_point)));
            checkCuda(cudaMemcpy(mg_ep_v[g], pv.data(), pv.size() * sizeof(exitation_point), cudaMemcpyHostToDevice));
        }
        if (!pa.empty())
        {
            checkCuda(cudaMalloc(&mg_ep_a[g], pa.size() * sizeof(exitation_point)));
            checkCuda(cudaMemcpy(mg_ep_a[g], pa.data(), pa.size() * sizeof(exitation_point), cudaMemcpyHostToDevice));
        }
        if (!pv.empty() || !pa.empty())
        {
            checkCuda(cudaMalloc(&mg_sig_v[g], length * sizeof(FDTD_FLOAT)));
            checkCuda(cudaMalloc(&mg_sig_a[g], length * sizeof(FDTD_FLOAT)));
            checkCuda(cudaMemcpy(mg_sig_v[g], exc_volt, length * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(mg_sig_a[g], exc_curr, length * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
        }
    }
}

void Engine_Ext_Excitation::Apply2VoltagesCuda(Engine_cuda *eng)
{
    if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
        Apply2VoltagesMg(mg);
        return;
    }
    int N = m_Op_Exc->Volt_Count;
    if (N <= 0 || this->d_ep_v == NULL)     return;

    // Constant across the run; 0 means "non-periodic" (kernel derives p=numTS+1).
    int signalPeriod = 0;
    if (m_Op_Exc->m_Exc->GetSignalPeriod() > 0) {
        signalPeriod = int(m_Op_Exc->m_Exc->GetSignalPeriod() / m_Op_Exc->m_Exc->GetTimestep());
    }

    int threads = std::min(1024, N);
    int blocks = (N + threads - 1) / threads;


    kernelApply2VA<<<blocks, threads>>>(
        eng->GetDeviceVoltData(),
        this->d_ep_v,
        this->d_signal_v,
        eng->GetDeviceDimData(),
        eng->GetDeviceNumTS(),
        signalPeriod,
        N
    );
    //checkCudaErrors();
}


void Engine_Ext_Excitation::Apply2CurrentCuda(Engine_cuda *eng)
{
    if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
        Apply2CurrentMg(mg);
        return;
    }
    int N = m_Op_Exc->Curr_Count;
    if (N <= 0 || this->d_ep_a == NULL)     return;

    int signalPeriod = 0;
    if (m_Op_Exc->m_Exc->GetSignalPeriod() > 0) {
        signalPeriod = int(m_Op_Exc->m_Exc->GetSignalPeriod() / m_Op_Exc->m_Exc->GetTimestep());
    }

    int threads = std::min(1024, N);
    int blocks = (N + threads - 1) / threads;

    kernelApply2VA<<<blocks, threads>>>(
        eng->GetDeviceCurrData(),
        this->d_ep_a,
        this->d_signal_a,
        eng->GetDeviceDimData(),
        eng->GetDeviceNumTS(),
        signalPeriod,
        N
    );

    //checkCudaErrors();
}





void  Engine_Ext_Excitation::SetEngine(Engine* eng)
{
    m_Eng = eng;

    if (eng->GetType() != Engine::CUDA) {
        return;
    }

    if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
        SetEngineMg(mg);
        return;
    }

    int n_volt = m_Op_Exc->Volt_Count;
    int n_curr = m_Op_Exc->Curr_Count;

    if (n_volt + n_curr <= 0) {
        return;
    }

	unsigned int length = m_Op_Exc->m_Exc->GetLength();
	FDTD_FLOAT* exc_volt =  m_Op_Exc->m_Exc->GetVoltageSignal();
	FDTD_FLOAT* exc_curr =  m_Op_Exc->m_Exc->GetCurrentSignal();

    exitation_point *ep_all = new exitation_point[n_volt + n_curr];

    // Per-point spew scales with excitation size (thousands for a plane-wave /
    // TFSF source) -- gate it behind OPENEMS_DEBUG_EXC so startup stays quiet.
    bool _dbg_exc = getenv("OPENEMS_DEBUG_EXC");
    if (_dbg_exc)
        printf(">>>>>>>>>>>>>>> load excitation signal to device memory: %d,%d\n", n_volt, n_curr);
    exitation_point *ep = ep_all;
    for (int i = 0; i < n_volt; i++) {
        ep->delay = m_Op_Exc->Volt_delay[i];
        ep->dir = m_Op_Exc->Volt_dir[i];
        ep->amp = m_Op_Exc->Volt_amp[i];
        ep->x = m_Op_Exc->Volt_index[0][i];
        ep->y = m_Op_Exc->Volt_index[1][i];
        ep->z = m_Op_Exc->Volt_index[2][i];
        ep->length = length;

        if (_dbg_exc)
            printf("load volt signal: %d,%d,%d, %f\n", ep->x, ep->y, ep->z, ep->amp);

        ep++;
    }
    for (int i = 0; i < n_curr; i++) {
        ep->delay = m_Op_Exc->Curr_delay[i];
        ep->dir = m_Op_Exc->Curr_dir[i];
        ep->amp = m_Op_Exc->Curr_amp[i];
        ep->x = m_Op_Exc->Curr_index[0][i];
        ep->y = m_Op_Exc->Curr_index[1][i];
        ep->z = m_Op_Exc->Curr_index[2][i];
        ep->length = length;

        if (_dbg_exc)
            printf("load curr signal: %d,%d,%d, %f\n", ep->x, ep->y, ep->z, ep->amp);

        ep++;
    }

    // not sure this is a good practice, by combining two arrays to avoid too much cudaMalloc calls.
    if (n_volt > 0) {
        checkCuda(cudaMalloc(&d_ep_v, sizeof(exitation_point) * n_volt));
        checkCuda(cudaMemcpy(d_ep_v, ep_all, sizeof(exitation_point) * n_volt, cudaMemcpyHostToDevice));
    }

    if (n_curr > 0) {
        checkCuda(cudaMalloc(&d_ep_a, sizeof(exitation_point) * n_curr));
        checkCuda(cudaMemcpy(d_ep_a, ep_all + n_volt, sizeof(exitation_point) * n_curr, cudaMemcpyHostToDevice));
    }


    checkCuda(cudaMalloc(&d_signal_v, sizeof(FDTD_FLOAT) * length));
    checkCuda(cudaMalloc(&d_signal_a, sizeof(FDTD_FLOAT) * length));


    checkCuda(cudaMemcpy(d_signal_v, exc_volt, sizeof(FDTD_FLOAT) * length, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_signal_a, exc_curr, sizeof(FDTD_FLOAT) * length, cudaMemcpyHostToDevice));

    delete ep_all;
}

#endif