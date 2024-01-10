#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2023-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

export starpudir=$PWD
export rootdir=$PWD/../starpu_chameleon
export builddir=$PWD/../starpu_chameleon/build
rm -rf $rootdir
mkdir -p $builddir

./autogen.sh
cd $builddir
$starpudir/configure --prefix=$rootdir/starpu.inst --disable-static --disable-socl --disable-build-tests --disable-build-examples --disable-build-doc --disable-opencl
make -j 32
make install
source $rootdir/starpu.inst/bin/starpu_env

# compiling morse
cd $rootdir
git clone --quiet --recursive --branch master https://gitlab.inria.fr/solverstack/chameleon.git chameleon
cd chameleon
git show HEAD
mkdir build
cd build
CFLAGS=-g cmake ../ -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON
make -j 20

set +e
ctest -R test_mpi_s
if test $? -ne 0
then
    ctest --rerun-failed --output-on-failure
fi
#ctest -R test_mpi_sgeadd -V
