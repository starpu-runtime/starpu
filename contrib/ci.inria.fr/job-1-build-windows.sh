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

oldPATH=$PATH

tarball=$(ls -tr starpu*.tar.gz | tail -1)
if test -z "$tarball" ; then
    echo Tarball not available
    exit 2
fi

basename=$(basename $tarball .tar.gz)
test -d $basename && chmod -R u+rwX $basename && rm -rf $basename
tar xfz $tarball
version=$(echo $basename | cut -d- -f2)
winball=starpu-win32-build-${version}

export STARPU_HOME=$PWD

rm -rf ${basename}/build
mkdir ${basename}/build
cd ${basename}/build

export PATH=/c/Builds:/c/MinGW/bin/:/c/MinGW/mingw32/bin/:/c/MinGW/msys/1.0/bin:"/c/Program Files (x86)/Microsoft Visual Studio 11.0/VC/bin":"/c/Program Files/Microsoft Visual Studio 11.0/Common7/IDE":$oldPATH
#export HWLOC=/c/StarPU/hwloc-win32-build-1.11.0

prefix=${PWD}/../../${winball}
rm -rf $prefix

day=$(date +%u)
if test $day -le 5
then
#    ../configure --prefix=$prefix --with-hwloc=${HWLOC} --disable-build-examples --enable-quick-check --enable-debug --enable-verbose --enable-native-winthreads
    ../configure --prefix=$prefix --without-hwloc --disable-build-examples --enable-quick-check --enable-debug --enable-verbose --enable-native-winthreads
else
#    ../configure --prefix=$prefix --with-hwloc=${HWLOC} --enable-quick-check --enable-debug --enable-verbose --enable-native-winthreads
    ../configure --prefix=$prefix --without-hwloc --enable-quick-check --enable-debug --enable-verbose --enable-native-winthreads
fi

make

CHECK=${PWD}/check_$$
touch ${CHECK}

if test "$1" == "-exec"
then
    (make -k check || true) > ${CHECK} 2>&1
    cat ${CHECK}
    make showcheck
fi

fail=$(grep FAIL ${CHECK} | grep -v XFAIL || true)
if test -z "$fail"
then
    make install
    cd ../../
    cp /c/MinGW/bin/pthread*dll ${winball}/bin
    cp /c/MinGW/bin/libgcc*dll ${winball}/bin
    #    cp ${HWLOC}/bin/*dll ${winball}/bin
    zip -r ${winball}.zip ${winball}

    rm -rf starpu_install
    mv ${winball} starpu_install
fi

PATH=$oldPATH

echo $fail
exit $(grep FAIL ${CHECK} | grep -v XFAIL | wc -l)

