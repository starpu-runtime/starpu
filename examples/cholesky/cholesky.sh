#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

ROOT=${0%.sh}
[ -z "$STARPU_SCHED" ] || STARPU_SCHEDS="$STARPU_SCHED"
#[ -n "$STARPU_SCHEDS" ] || STARPU_SCHEDS=`$(dirname $0)/../../tools/starpu_sched_display`
[ -n "$STARPU_SCHEDS" ] || STARPU_SCHEDS="dmdas modular-heft2 modular-heft modular-heft-prio modular-heteroprio dmdap dmdar dmda dmdasd prio lws"
[ -n "$STARPU_HOSTNAME" ] || export STARPU_HOSTNAME=mirage
unset MALLOC_PERTURB_

INCR=2
STOP=32

if [ -n "$STARPU_SIMGRID" ]
then
	INCR=4
	STOP=14
	# These use the thread factory, and are thus much longer
	if [ -n "$STARPU_QUICK_CHECK" ]
	then
		INCR=8
		STOP=10
	fi
	if [ -n "$STARPU_LONG_CHECK" ]
	then
		INCR=4
		STOP=32
	fi
fi

(
echo -n "#"
for STARPU_SCHED in $STARPU_SCHEDS ; do
	echo -n "	$STARPU_SCHED"
done
echo

for size in `seq 2 $INCR $STOP` ; do
	echo -n "$((size * 960))"
	for STARPU_SCHED in $STARPU_SCHEDS
	do
		export STARPU_SCHED
		GFLOPS=`$STARPU_LAUNCH ${ROOT}_implicit -size $((size * 960)) -nblocks $size 2> /dev/null | grep -v GFlops | cut -d '	' -f 3`
		[ -n "$GFLOPS" ] || GFLOPS='""'
		echo -n "	$GFLOPS"
	done
	echo 
done
) | tee cholesky.output

[ -n "$TERMINAL" ] || TERMINAL=eps
[ -n "$OUTFILE" ] || OUTFILE=cholesky.eps
cat > cholesky.gp << EOF
set terminal $TERMINAL
set output "$OUTFILE"
set key top left
set xlabel "size"
set ylabel "GFlops"
plot \\
EOF

N=2
COMMA=""
for STARPU_SCHED in $STARPU_SCHEDS
do
	echo "$COMMA'cholesky.output' using 1:$N with lines title '$STARPU_SCHED' \\" >> cholesky.gp
	N=$(($N + 1))
	COMMA=", "
done
gnuplot cholesky.gp
#gv $OUTFILE
true
