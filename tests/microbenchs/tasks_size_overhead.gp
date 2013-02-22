#!/bin/sh

# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009, 2010  Université de Bordeaux 1
# Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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


#!/bin/sh
OUTPUT=tasks_size_overhead.output
VALS=$(sed -n -e '3p' < $OUTPUT)
VAL1=$(echo "$VALS" | cut -d '	' -f 3)
VAL2=$(echo "$VALS" | cut -d '	' -f 5)
VAL3=$(echo "$VALS" | cut -d '	' -f 7)
VAL4=$(echo "$VALS" | cut -d '	' -f 9)
VAL5=$(echo "$VALS" | cut -d '	' -f 11)
VAL6=$(echo "$VALS" | cut -d '	' -f 13)
VAL7=$(echo "$VALS" | cut -d '	' -f 15)
VAL8=$(echo "$VALS" | cut -d '	' -f 17)
VAL9=$(echo "$VALS" | cut -d '	' -f 19)
VAL10=$(echo "$VALS" | cut -d '	' -f 21)
VAL11=$(echo "$VALS" | cut -d '	' -f 23)
gnuplot << EOF
set terminal eps
set output "tasks_size_overhead.eps"
set key top left
set xlabel "number of cores"
set ylabel "speedup"
plot \
	"$OUTPUT" using 1:($VAL1)/(\$3) with linespoints title columnheader(2), \
	"$OUTPUT" using 1:($VAL2)/(\$5) with linespoints title columnheader(4), \
	"$OUTPUT" using 1:($VAL3)/(\$7) with linespoints title columnheader(6), \
	"$OUTPUT" using 1:($VAL4)/(\$9) with linespoints title columnheader(8), \
	"$OUTPUT" using 1:($VAL5)/(\$11) with linespoints title columnheader(10), \
	"$OUTPUT" using 1:($VAL6)/(\$13) with linespoints title columnheader(12), \
	"$OUTPUT" using 1:($VAL7)/(\$15) with linespoints title columnheader(14), \
	"$OUTPUT" using 1:($VAL8)/(\$17) with linespoints title columnheader(16), \
	"$OUTPUT" using 1:($VAL9)/(\$19) with linespoints title columnheader(18), \
	"$OUTPUT" using 1:($VAL10)/(\$21) with linespoints title columnheader(20), \
	"$OUTPUT" using 1:($VAL11)/(\$23) with linespoints title columnheader(22), \
	x title "linear"
EOF
