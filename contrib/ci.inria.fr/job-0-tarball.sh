#!/bin/sh

export PKG_CONFIG_PATH=/home/ci/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/ci/usr/local/lib:$LD_LIBRARY_PATH

./autogen.sh
mkdir build && cd build
../configure
make V=1
make dist
make clean

