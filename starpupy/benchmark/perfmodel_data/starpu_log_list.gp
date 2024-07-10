#!/usr/bin/gnuplot -persist
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022-2024   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
set output "starpu_log_list.eps"
set title "Model for codelet log-list on he-XPS-13-9370"
set xlabel "Total data size"
set ylabel "Time (ms)"

set key top left
set logscale x
set logscale y

set xrange [1 < * < 10**5 : 10**6 < * < 10**9]

plot	".//starpu_log_list_avg.data" using 1:2:3 with errorlines title "Average cpu0-impl0 (Comb0)"
