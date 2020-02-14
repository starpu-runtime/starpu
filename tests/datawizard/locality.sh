#!/bin/sh -x
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017                                     CNRS
# Copyright (C) 2017                                     Inria
# Copyright (C) 2017,2018-2020                           Université de Bordeaux
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
# Test generation of FxT traces

set -e

PREFIX=$(dirname $0)
test -x $PREFIX/../../tools/starpu_fxt_tool || exit 77
STARPU_SCHED=modular-eager STARPU_FXT_PREFIX=$PREFIX/ $PREFIX/locality
$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_tool -i $PREFIX/prof_file_${USER}_0

# Check that they are approved by Grenoble :)

if type pj_dump > /dev/null 2> /dev/null
then
	$PREFIX/../../tools/starpu_paje_sort paje.trace
	pj_dump paje.trace
fi
