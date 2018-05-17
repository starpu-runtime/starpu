#!/bin/sh

export PKG_CONFIG_PATH=/home/ci/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/ci/usr/local/lib:$LD_LIBRARY_PATH

./autogen.sh
if test -d build ; then chmod -R 777 build && rm -rf build ; fi
mkdir build && cd build
../configure
make V=1
make dist
cp doc/doxygen/starpu.pdf ..
make clean

