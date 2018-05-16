#!/bin/sh

./autogen.sh
mkdir build && cd build
../configure
make
make distcheck

