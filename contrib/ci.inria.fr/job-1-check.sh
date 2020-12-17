#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

ulimit -c unlimited

export PKG_CONFIG_PATH=/home/ci/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/ci/usr/local/lib:$LD_LIBRARY_PATH

tarball=$(ls -tr starpu-*.tar.gz | tail -1)

if test -z "$tarball"
then
    echo Error. No tar.gz file
    ls
    pwd
    exit 1
fi

basename=$(basename $tarball .tar.gz)
export STARPU_HOME=$PWD/$basename/home
mkdir -p $basename
cd $basename
env > $PWD/env

test -d $basename && chmod -R u+rwX $basename && rm -rf $basename
tar xfz ../$tarball
cd $basename
touch configure
mkdir build
cd build

STARPU_CONFIGURE_OPTIONS=""
suname=$(uname)
if test "$suname" = "Darwin"
then
    STARPU_CONFIGURE_OPTIONS="--without-hwloc"
fi
if test "$suname" = "OpenBSD"
then
    STARPU_CONFIGURE_OPTIONS="--without-hwloc --disable-mlr"
fi
if test "$suname" = "FreeBSD"
then
    STARPU_CONFIGURE_OPTIONS="--disable-fortran"
fi

export CC=gcc

CONFIGURE_OPTIONS="--enable-debug --enable-verbose --enable-mpi-check --disable-build-doc"
CONFIGURE_CHECK=""
day=$(date +%u)
if test $day -le 5
then
    CONFIGURE_CHECK="--enable-quick-check"
#else
    # we do a normal check, a long check takes too long on VM nodes
fi
../configure $CONFIGURE_OPTIONS $CONFIGURE_CHECK  $STARPU_CONFIGURE_OPTIONS

export STARPU_TIMEOUT_ENV=1800
make
#make check
(make -k check || true) 2>&1 | tee  ../check_$$
make showsuite

grep "^FAIL:" ../check_$$ || true

make clean

grep "^FAIL:" ../check_$$ || true

echo "Running on $(uname -a)"
exit $(grep "^FAIL:" ../check_$$ | wc -l)

