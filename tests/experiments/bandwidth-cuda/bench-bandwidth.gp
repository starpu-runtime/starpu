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
set output "bench-bandwidth.eps"
set title "CUDA Bandwidth"
set logscale x
set xlabel "Size (Bytes)"
set ylabel "Bandwidth (MB/s)"

plot ".results/htod-pin.data" with linespoint	title "Host to Device (pinned)",\
     ".results/dtoh-pin.data" with linespoint   title "Device to Host (pinned)"

