#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
# Test various LU options

set -e

PREFIX=$(dirname $0)
rm -rf $PREFIX/lu.traces
mkdir -p $PREFIX/lu.traces

export STARPU_FXT_PREFIX=$PREFIX/lu.traces
export STARPU_FXT_TRACE=1

if [ "$STARPU_QUICK_CHECK" = 1 ]
then
	SIDE=16
else
	SIDE=160
fi

$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $(($SIDE * 4)) -nblocks 4 -piv
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $(($SIDE * 4)) -nblocks 4 -no-stride
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $(($SIDE * 4)) -nblocks 4 -bound
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $(($SIDE * 2)) -nblocks 2 -bounddeps -directory $STARPU_FXT_PREFIX
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $(($SIDE * 2)) -nblocks 2 -bound -bounddeps -bounddepsprio -directory $STARPU_FXT_PREFIX

$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_example_float -size $(($SIDE * 4)) -nblocks 4 -piv
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_example_float -size $(($SIDE * 4)) -nblocks 4 -no-stride
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_example_float -size $(($SIDE * 4)) -nblocks 4 -bound
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_example_float -size $(($SIDE * 2)) -nblocks 2 -bounddeps -directory $PREFIX/lu.traces
$MS_LAUNCHER $STARPU_LAUNCH $PREFIX/lu_example_float -size $(($SIDE * 2)) -nblocks 2 -bound -bounddeps -bounddepsprio -directory $STARPU_FXT_PREFIX
