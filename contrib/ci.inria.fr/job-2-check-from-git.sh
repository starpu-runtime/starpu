#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set -e
set -x

echo "Running on $(uname -a)"

export LC_ALL=C
ulimit -c unlimited

export PKG_CONFIG_PATH=/home/ci/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/ci/usr/local/lib:$LD_LIBRARY_PATH

if test -f $HOME/starpu_specific_env.sh
then
    . $HOME/starpu_specific_env.sh
fi

export CC=gcc
export STARPU_MICROBENCHS_DISABLED=1
export STARPU_TIMEOUT_ENV=3600
export MPIEXEC_TIMEOUT=3600
CONFIGURE_OPTIONS="--enable-debug --enable-verbose --enable-mpi-check=maybe --enable-mpi-minimal-tests --disable-build-doc --enable-quick-check"

set +e
mpiexec -oversubscribe pwd 2>/dev/null
ret=$?
set -e
ARGS=""
if test "$ret" = "0"
then
    ARGS="--with-mpiexec-args=-oversubscribe"
fi

./autogen.sh

BUILD=./build_$$
if test -d $BUILD ; then chmod -R 777 $BUILD && rm -rf $BUILD ; fi
mkdir $BUILD && cd $BUILD
../configure $CONFIGURE_OPTIONS $ARGS $STARPU_USER_CONFIGURE_OPTIONS
make -j4
set +e
set -o pipefail
make -k check 2>&1 | tee  ../check_$$
RET=$?
make showcheckfailed
make clean
grep "^FAIL:" ../check_$$ || true

if test "$RET" != "0"
then
    STARPU_USER_CONFIGURE_OPTIONS="--enable-simgrid --disable-cuda --disable-mpi --disable-mpi-check"
    BUILD=./build_simgrid_$$
    if test -d $BUILD ; then chmod -R 777 $BUILD && rm -rf $BUILD ; fi
    mkdir $BUILD && cd $BUILD
    ../configure $CONFIGURE_OPTIONS $ARGS $STARPU_USER_CONFIGURE_OPTIONS
    make -j4
    set +e
    set -o pipefail
    make -k check 2>&1 | tee  ../check_$$
    RET=$?
    make showcheckfailed
    make clean
    grep "^FAIL:" ../check_$$ || true
fi

exit $RET
