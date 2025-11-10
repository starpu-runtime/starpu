#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

export LC_ALL=C
export PKG_CONFIG_PATH=/home/ci/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/ci/usr/local/lib:$LD_LIBRARY_PATH

if test -f $HOME/starpu_specific_env.sh
then
    . $HOME/starpu_specific_env.sh
fi

configure_doc="--enable-build-doc-pdf"
if test "$1" == "--disable-doc"
then
    configure_doc=""
fi

BUILD=./build_$$
./autogen.sh
if test -d $BUILD ; then chmod -R 777 $BUILD && rm -rf $BUILD ; fi
mkdir $BUILD && cd $BUILD
../configure $configure_doc $STARPU_USER_CONFIGURE_OPTIONS
make -j4
make dist
cp *gz ..
if test "$1" != "--disable-doc"
then
    cp doc/doxygen/starpu.pdf ..
    cp doc/doxygen_dev/starpu_dev.pdf ..
    cp -rp doc/doxygen/html ..
fi
make clean
cd ../

tarball=$(ls -tr starpu-*.tar.gz | tail -1)
if test -z "$tarball"
then
    echo Error. No tar.gz file
    ls
    pwd
    exit 1
fi

if test "$1" != "--disable-doc"
then
    if test ! -f starpu.pdf
    then
	echo Error. No documentation file
	ls
	pwd
	exit 1
    fi
fi
