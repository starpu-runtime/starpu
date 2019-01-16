#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017                                     CNRS
# Copyright (C) 2017,2019                                Université de Bordeaux
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

$PREFIX/lu_implicit_example_float -size $((160 * 4)) -nblocks 4 -piv
$PREFIX/lu_implicit_example_float -size $((160 * 4)) -nblocks 4 -no-stride
$PREFIX/lu_implicit_example_float -size $((160 * 4)) -nblocks 4 -bound
$PREFIX/lu_implicit_example_float -size $((160 * 2)) -nblocks 2 -bounddeps
$PREFIX/lu_implicit_example_float -size $((160 * 2)) -nblocks 2 -bound -bounddeps -bounddepsprio

$PREFIX/lu_example_float -size $((160 * 4)) -nblocks 4 -piv
$PREFIX/lu_example_float -size $((160 * 4)) -nblocks 4 -no-stride
$PREFIX/lu_example_float -size $((160 * 4)) -nblocks 4 -bound
$PREFIX/lu_example_float -size $((160 * 2)) -nblocks 2 -bounddeps
$PREFIX/lu_example_float -size $((160 * 2)) -nblocks 2 -bound -bounddeps -bounddepsprio
