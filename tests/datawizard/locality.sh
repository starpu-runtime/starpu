#!/bin/sh -x
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2018       Federal University of Rio Grande do Sul (UFRGS)
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

# Testing another specific scheduler, no need to run this
[ -z "$STARPU_SCHED" -o "$STARPU_SCHED" = modular-eager ] || exit 77

set -e

PREFIX=$(dirname $0)
rm -rf $PREFIX/locality.traces
mkdir -p $PREFIX/locality.traces

test -x $PREFIX/../../tools/starpu_fxt_tool || exit 77

export STARPU_FXT_PREFIX=$PREFIX/locality.traces
STARPU_FXT_TRACE=1 STARPU_SCHED=modular-eager $MS_LAUNCHER $STARPU_LAUNCH $PREFIX/locality
$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_tool -d $STARPU_FXT_PREFIX -memory-states -label-deps -i $STARPU_FXT_PREFIX/prof_file_${USER}_0

# Check that they are approved by Grenoble :)

if type pj_dump > /dev/null 2> /dev/null
then
	$PREFIX/../../tools/starpu_paje_sort $STARPU_FXT_PREFIX/paje.trace
	pj_dump -e 0 $STARPU_FXT_PREFIX/paje.trace
fi
