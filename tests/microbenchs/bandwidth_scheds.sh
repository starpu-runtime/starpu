#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set -e

if [ -n "$STARPU_SCHED" ]
then
	SCHEDS=$STARPU_SCHED
	DEFAULT=$STARPU_SCHED
else
	SCHEDS=`$(dirname $0)/../../tools/starpu_sched_display`
	DEFAULT=eager
fi

if [ -n "$STARPU_BENCH_DIR" ]; then
	cat > bandwidth.gp << EOF
set term svg font ",12" size 1500,500 linewidth 0.5
set output "bandwidth.svg"
set pointsize 0.3
EOF
else
	fast="-n 3 -c 4"
	cat > bandwidth.gp << EOF
set term postscript eps enhanced color font ",18"
set output "bandwidth.eps"
set size 2,1
EOF
fi

cat >> bandwidth.gp << EOF
set key outside
set ylabel "GB/s"
set xlabel "ncores"

plot \\
	"bandwidth-$DEFAULT.dat" using 1:5 with lines title "alone interleave", \\
	"bandwidth-$DEFAULT.dat" using 1:6 with lines title "nop", \\
	"bandwidth-$DEFAULT.dat" using 1:7 with lines title "sync", \\
	"bandwidth-$DEFAULT.dat" using 1:2 with lines title "alone contiguous", \\
EOF

type=1
for sched in $SCHEDS
do
	if [ "$sched" != eager -a "$sched" != "$SCHEDS" ]; then
		extra=-a
	else
		extra=
	fi

	STARPU_BACKOFF_MIN=0 STARPU_BACKOFF_MAX=0 STARPU_SCHED=$sched $STARPU_LAUNCH $(dirname $0)/bandwidth $fast $extra "$@" | tee bandwidth-$sched.dat
	echo "\"bandwidth-$sched.dat\" using 1:3 with linespoints lt $type pt $type title \"$sched\", \\" >> bandwidth.gp
	echo "\"bandwidth-$sched.dat\" using 1:8 with linespoints lt $type pt $type notitle, \\" >> bandwidth.gp
	type=$((type+1))
done

if gnuplot bandwidth.gp ; then
	if [ -n "$STARPU_BENCH_DIR" ]; then
		cp bandwidth.svg $STARPU_BENCH_DIR/
	fi
fi
