#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

export STARPU_MICROBENCHS_DISABLED=1
export STARPU_CHECK_DIRS=mpi
export STARPU_USER_CONFIGURE_OPTIONS="--with-mpicc=/usr/bin/mpicc.mpich --with-mpiexec=/usr/bin/mpiexec.mpich --with-mpicxx=/usr/bin/mpicxx.mpich --with-mpifort=/usr/bin/mpifort.mpich --disable-mpi-type-vector-c"
./contrib/ci.inria.fr/job-1-check.sh
