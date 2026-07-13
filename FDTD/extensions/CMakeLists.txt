
#INCLUDE_DIRECTORIES( ${openEMS_SOURCE_DIR} )
#INCLUDE_DIRECTORIES( ${CSXCAD_SOURCE_DIR}/src )

if (WITH_CUDA)
  set(CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_excitation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_united_upml.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_united_upml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_mur_abc.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_lumpedRLC.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_lorentzmaterial.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_tfsf.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_steadystate.cu
  )
endif()


set(SOURCES
  ${SOURCES}
  ${CUDA_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_extension.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_dispersive.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_lorentzmaterial.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_conductingsheet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_dispersive.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_lorentzmaterial.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_cylindermultigrid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_upml.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_upml.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_extension.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_mur_abc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_mur_abc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_cylinder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_cylinder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_excitation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_excitation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_tfsf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_tfsf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_steadystate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_steadystate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_ext_lumpedRLC.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_lumpedRLC.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/engine_ext_united_upml.cpp
  PARENT_SCOPE
)

# FDTD/extensions lib
#add_library( extensions STATIC ${SOURCES} )

