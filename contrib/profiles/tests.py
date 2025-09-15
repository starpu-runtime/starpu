# StarPU --- Runtime system for heterogeneous multicore architectures.
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2025  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

config = {
    'opts'        : [ '--disable-static', '--disable-build-doc', '--enable-debug', '--enable-starpupy', '--enable-hdf5'],
    'env'         : {'STARPU_SILENT' : '1',
                     'STARPU_SSILENT' : '1',
                     'SOCL_OCL_LIB_OPENCL' : '/usr/lib/x86_64-linux-gnu/libOpenCL.so',
                     'MALLOC_PERTURB_' : '1234',
                     'OPENBLAS_NUM_THREADS': '1'
                    },
}

profiles = [
    # coverage
    { 'name'        : 'coverage',
      'release'     : True,
      'deploy'      : True,
      'coverage'    : ['--enable-coverage'],
      'opts'        : ['--enable-build-doc', '--enable-build-doc-pdf'],
      'hosts'       : ['node_release'],
      'ignore_fail' : True
    },
]

