#!/usr/bin/gnuplot -persist

#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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
set output "model_error.eps"

set yrange [0.05:100]
set xrange [4:16380]

set grid y
set grid x

set logscale y
#set logscale x

#set ytics 1, 2.5, 50

set ytics (0.01, 0.1, 1, 5,10,25,50)
set xtics (10, 100, 1000, 10000)

set format y "%.2f %%"

plot "gnuplot.data" usi 3:($4*100) with linespoint title "prediction error (GPU)"	,\
     "gnuplot.data" usi 1:($2*100) with linespoint title "prediction error (CPUs)"
