# -*- coding: utf-8 -*-
#
# Shortcut openEMS import
from __future__ import absolute_import

# try to import CSXCAD, e.g. make sure the (windows) dll path are set
import CSXCAD

from openEMS.openEMS import openEMS

# Capabilities of this build that callers may branch on (compute deploys
# independently of the engine image, so it checks instead of assuming):
#   mgpu_field_dft -- the multi-GPU CUDA engine gathers field dumps and runs
#                     their frequency-domain DFT on the devices, as fast as
#                     one GPU per slab (before, dumps forced a slow host path).
FEATURES = frozenset(["mgpu_field_dft"])
