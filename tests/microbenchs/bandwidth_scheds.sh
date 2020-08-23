#!/bin/sh
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
	set term png font ",16"
	set output "bandwidth.png"
EOF
else
	cat > bandwidth.gp << EOF
	set term postscript eps enhanced color font ",18"
	set output "bandwidth.eps"
EOF
fi

cat >> bandwidth.gp << EOF
set key outside
set ylabel "GB/s"
set xlabel "ncores"

plot \\
	"bandwidth-$DEFAULT.dat" using 1:2 with lines title "alone", \\
EOF

for sched in $SCHEDS
do
	if [ "$sched" != eager -a "$sched" != "$SCHEDS" ]; then
		extra=-a
	else
		extra=
	fi

	STARPU_BACKOFF_MIN=0 STARPU_BACKOFF_MAX=0 STARPU_SCHED=$sched $STARPU_LAUNCH $(dirname $0)/bandwidth $extra | tee bandwidth-$sched.dat
	echo "\"bandwidth-$sched.dat\" using 1:3 with linespoints title \"$sched\", \\" >> bandwidth.gp
done

if gnuplot bandwidth.gp ; then
	if [ -n "$STARPU_BENCH_DIR" ]; then
		cp bandwidth.png $STARPU_BENCH_DIR/
	fi
fi
