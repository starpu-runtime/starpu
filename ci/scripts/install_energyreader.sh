#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2022, 2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

cur_dir=${PWD}

mkdir -p ${starpu_build_dir}/../energyreader
XROOT=$(realpath ${starpu_build_dir}/../energyreader)
cd ${XROOT}

rm -rf energy-reader
git clone https://gitlab.com/JulesRisse/energy-reader
cd energy-reader
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$XROOT/install
make install

export PKG_CONFIG_PATH=$XROOT/install/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$XROOT/install/lib:$LD_LIBRARY_PATH

cd $cur_dir
