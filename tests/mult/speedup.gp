#!/usr/bin/gnuplot -persist

#
# StarPU
# Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#


set term postscript eps enhanced color
set output "speedup.eps"
set title "Matrix multiplication"
set xlabel "number of CPUs"
set ylabel "speedup"
plot 	"timings/speedup.4096" usi 1:1	 with lines title "ideal",\
	"timings/speedup.2048" usi 1:($3/$2) with lines title "2048",\
	"timings/speedup.4096" usi 1:($3/$2) with lines title "4096",\
	"timings/speedup.8192" usi 1:($3/$2) with lines title "8192"
