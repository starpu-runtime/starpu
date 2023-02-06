#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

if test -n "$STARPU_MICROBENCHS_DISABLED" ; then exit 77 ; fi

source $(dirname $0)/microbench.sh

XFAIL="lws ws eager prio modular-prio modular-eager modular-eager-prio modular-eager-prefetching modular-prio-prefetching modular-random modular-random-prio modular-random-prefetching modular-random-prio-prefetching modular-prandom modular-prandom-prio modular-ws modular-heft modular-heft-prio modular-heft2 modular-heteroprio modular-gemm random peager heteroprio graph_test"

test_scheds parallel_independent_heterogeneous_tasks
