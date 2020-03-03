#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

if [ -n "$STARPU_MIC_SINK_PROGRAM_PATH" ] ; then
	STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/lu_implicit_example_float
	# in case libtool got into play
	[ -x "$STARPU_MIC_SINK_PROGRAM_PATH/.libs/lu_implicit_example_float" ] && STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/.libs/lu_implicit_example_float
fi

$STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $((160 * 4)) -nblocks 4 -piv
$STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $((160 * 4)) -nblocks 4 -no-stride
$STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $((160 * 4)) -nblocks 4 -bound
$STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $((160 * 2)) -nblocks 2 -bounddeps
$STARPU_LAUNCH $PREFIX/lu_implicit_example_float -size $((160 * 2)) -nblocks 2 -bound -bounddeps -bounddepsprio

if [ -n "$STARPU_MIC_SINK_PROGRAM_PATH" ] ; then
	STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/lu_example_float
	# in case libtool got into play
	[ -x "$STARPU_MIC_SINK_PROGRAM_PATH/.libs/lu_example_float" ] && STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/.libs/lu_example_float
fi

$STARPU_LAUNCH $PREFIX/lu_example_float -size $((160 * 4)) -nblocks 4 -piv
$STARPU_LAUNCH $PREFIX/lu_example_float -size $((160 * 4)) -nblocks 4 -no-stride
$STARPU_LAUNCH $PREFIX/lu_example_float -size $((160 * 4)) -nblocks 4 -bound
$STARPU_LAUNCH $PREFIX/lu_example_float -size $((160 * 2)) -nblocks 2 -bounddeps
$STARPU_LAUNCH $PREFIX/lu_example_float -size $((160 * 2)) -nblocks 2 -bound -bounddeps -bounddepsprio
