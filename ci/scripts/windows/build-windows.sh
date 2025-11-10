#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

export LC_ALL=C
oldPATH=$PATH
# Add both PATHS for msys and cygdrive
export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/bin":"/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE":"/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64":$oldPATH
export PATH="/cygdrive/c/Program Files/Microsoft Visual Studio/2022/Community/VC/bin":"/cygdrive/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE":"/cygdrive/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64":$PATH

tarball=$(ls -tr starpu*.tar.gz | tail -1)
if test -z "$tarball" ; then
    echo Tarball not available
    exit 2
fi

basename=$(basename $tarball .tar.gz)
test -d $basename && chmod -R u+rwX $basename && rm -rf $basename
tar xfz $tarball
touch --date="last hour" $(find $basename)
version=$(echo $basename | cut -d- -f2)
winball=starpu-win64-build-${version}

export STARPU_HOME=$PWD
mkdir artifacts

rm -rf ${basename}/build
mkdir -p ${basename}/build
cd ${basename}/build

#export HWLOC=/c/StarPU/hwloc-win32-build-1.11.0

prefix=${STARPU_HOME}/${winball}
rm -rf $prefix

#--with-hwloc=${HWLOC}
options="--without-hwloc --enable-quick-check --enable-debug --enable-verbose"
#--enable-native-winthreads"
day=$(date +%u)
ret=0
set +e
if test $day -le 5
then
    ../configure --prefix=$prefix $options --disable-build-examples $STARPU_USER_CONFIGURE_OPTIONS
    ret=$?
else
    ../configure --prefix=$prefix $options $STARPU_USER_CONFIGURE_OPTIONS
    ret=$?
fi
set -e

# save config.log as artifact
cp config.log $STARPU_HOME/artifacts

# deal with configure error
if test "$ret" != "0"
then
    exit $ret
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
    cd $prefix/../
    #cp /c/MinGW/bin/pthread*dll ${winball}/bin
    #cp /c/MinGW/bin/libgcc*dll ${winball}/bin
    #    cp ${HWLOC}/bin/*dll ${winball}/bin
    zip -r ${winball}.zip ${winball}
    mv ${winball}.zip $STARPU_HOME/artifacts
fi

PATH=$oldPATH

echo $fail
exit $(grep FAIL ${CHECK} | grep -v XFAIL | wc -l)
