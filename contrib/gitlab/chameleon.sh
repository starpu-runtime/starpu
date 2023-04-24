#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set -x
set -e

export rootdir=$PWD/../starpu_chameleon
rm -rf $rootdir
mkdir -p $rootdir

./autogen.sh
mkdir build && cd build
../configure --prefix=$rootdir/starpu.inst --disable-static --disable-socl --disable-build-tests --disable-build-examples --disable-build-doc --disable-opencl
make -j 32
make install
source $rootdir/starpu.inst/bin/starpu_env

# compiling morse
cd $rootdir
rm -fr morse
mkdir morse
git clone --quiet --recursive --branch master https://gitlab.inria.fr/solverstack/chameleon.git morse/master
cd morse/master
mkdir build
cd build
CFLAGS=-g cmake ../ -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON
make -j 20

set +e
ctest -R test_mpi_s
ctest --rerun-failed --output-on-failure
#ctest -R test_mpi_sgeadd -V
