# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:48:22 2015

@author: thorsten
"""

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

import os, sys
import numpy
ROOT_DIR = os.path.dirname(__file__)

sys.path.append(os.path.join(ROOT_DIR,'..','..','CSXCAD','python'))

# Locate the C++ install prefix so headers/libs are found when openEMS is
# installed outside a system path (upstream setup.py does this; the fork's
# simplified version dropped it). rpath lets the .so self-locate at import.
_prefix = os.environ.get('OPENEMS_INSTALL_PATH') or os.environ.get('VIRTUAL_ENV')
_inc = [numpy.get_include()]
_lib = []
if _prefix:
    _inc.append(os.path.join(_prefix, 'include'))
    _lib.append(os.path.join(_prefix, 'lib'))

extensions = [
    Extension("*", [os.path.join(os.path.dirname(__file__), "openEMS","*.pyx")],
        language="c++",             # generate C++ code
        libraries    = ['CSXCAD','openEMS', 'nf2ff'],
        include_dirs = _inc,
        library_dirs = _lib,
        runtime_library_dirs = _lib),
]

setup(
  name="openEMS",
  version = '0.0.36',
  description = "Python interface for the openEMS FDTD library",
  classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Intended Audience :: Information Technology',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.9',
      'Programming Language :: Python :: 3.10',
      'Programming Language :: Python :: 3.11',
      'Programming Language :: Python :: 3.12',
      'Programming Language :: Python :: Implementation :: CPython',
      'Topic :: Scientific/Engineering',
      'Topic :: Software Development :: Libraries :: Python Modules',
      'Operating System :: POSIX :: Linux',
      'Operating System :: Microsoft :: Windows',
  ],
  author = 'Thorsten Liebig',
  author_email = 'Thorsten.Liebig@gmx.de',
  maintainer = 'Thorsten Liebig',
  maintainer_email = 'Thorsten.Liebig@gmx.de',
  url = 'https://openEMS.de',
  packages=["openEMS", ],
  package_data={'openEMS': ['*.pxd']},
  python_requires='>=3.9',
  install_requires=[
    'cython', # Apache License 2.0 (https://github.com/cython/cython/blob/master/LICENSE.txt)
    'h5py',   # BSD 3-Clause (https://github.com/h5py/h5py/blob/master/LICENSE)
    'numpy',  # BSD 3-Clause (https://github.com/numpy/numpy/blob/main/LICENSE.txt)
  ],
  ext_modules = cythonize(extensions, language_level = 3)
 )
