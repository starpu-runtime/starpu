#!/usr/bin/gnuplot -persist
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2008-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set term postscript eps enhanced color
set output "speedup.eps"
set title "Facto LU : impact of granularity"
set xlabel "problem size (unknowns)"
set ylabel "speedup ( 3 Cpus + 1 GPU vs . 4 Cpus )"
set yrange [0.5:2.5]
plot "speedup.8" usi 1:($3/$2) with lines title "(8x8)",\
	"speedup.16" usi 1:($3/$2) with lines title "(16x16)",\
	"speedup.32" usi 1:($3/$2) with lines title "(32x32)",\
	"speedup.8" usi 1:((1))	with lines title "Reference"
