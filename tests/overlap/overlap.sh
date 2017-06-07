#!/bin/bash -x
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017  Université de Bordeaux
# Copyright (C) 2017  Inria
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

# Test parsing of FxT traces

set -e

PREFIX=$(dirname $0)
test -x $PREFIX/../../tools/starpu_perfmodel_plot || exit 77
STARPU_SCHED=dmdas STARPU_FXT_PREFIX=$PREFIX/ $PREFIX/overlap
$PREFIX/../../tools/starpu_perfmodel_plot -s overlap_sleep_1024_24 -i $PREFIX/prof_file_${USER}_0
