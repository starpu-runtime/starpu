#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
# Testing another specific scheduler, no need to run this
[ -z "$STARPU_SCHED" -o "$STARPU_SCHED" = pheft ] || exit 77

#export STARPU_GENERATE_TRACE=1
#export GOMP_CPU_AFFINITY="0 6 1 7 2 8 3 9 4 10 5 11"
#export OMP_WAIT_POLICY=PASSIVE
export STARPU_SCHED=pheft
export STARPU_NCPUS=12
export STARPU_SINGLE_COMBINED_WORKER=1
export STARPU_MIN_WORKERSIZE=12
export STARPU_MAX_WORKERSIZE=12
export STARPU_NCUDA=0
export STARPU_NOPENCL=0
export STARPU_WORKER_STATS=1
export STARPU_CALIBRATE=1
$MS_LAUNCHER $STARPU_LAUNCH exec ./bfs data/graph65536.txt
