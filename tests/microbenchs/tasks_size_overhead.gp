#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

OUTPUT=tasks_size_overhead.output
VALS=$(sed -n -e '3p' < $OUTPUT)

PLOTS=""
for x in $(seq 1 11)
do
    pos=$((2 * $x + 1))
    double=$((2 * $x))
    value=$(echo "$VALS" | cut -d '	' -f $pos)
    if test -n "$value"
    then
	PLOTS=",\"$OUTPUT\" using 1:($value)/(\$$pos) with linespoints title columnheader($double) $PLOTS"
    fi
done

[ -n "$TERMINAL" ] || TERMINAL=eps
[ -n "$OUTFILE" ] || OUTFILE=tasks_size_overhead.eps
gnuplot << EOF
set terminal $TERMINAL
set output "$OUTFILE"
set key top left
set xlabel "number of cores"
set ylabel "speedup"
plot \
	x title "linear" $PLOTS
EOF

