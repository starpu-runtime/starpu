#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021, 2023       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
ROOT=${0%.sh}
for x in handle futur none
do
    $(dirname $0)/../execute.sh benchmark/tasks_size_overhead.py $x $*
    TERMINAL="png large size 1280,960" OUTFILE="tasks_size_overhead_py_$x.png" $ROOT.gp
    TERMINAL="eps" OUTFILE="tasks_size_overhead_py_$x.eps" $ROOT.gp
done
#gv tasks_size_overhead.eps
