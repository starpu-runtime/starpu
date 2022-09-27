#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2022       Camille Coti
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

ROOT=${0%/prof.sh}
if test -x $ROOT/../basic_examples/hello_world
then
    STARPU_PROF_TOOL=$ROOT/.libs/libprofiling_tool.so $ROOT/../basic_examples/hello_world
else
    exit 77
fi
