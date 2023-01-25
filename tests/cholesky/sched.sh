#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# Testing another specific scheduler, no need to run this
[ -z "$STARPU_SCHED" -o "$STARPU_SCHED" = greedy -o "$STARPU_SCHED" = no-prio -o "$STARPU_SCHED" = dm -o "$STARPU_SCHED" = random ] || exit 77

maxiter=10
MAXCPU=3

trace_sched()
{
	for blocks in `seq 10 2 24`
	do
		size=$(($blocks*1024))
	
		echo "size : $size"
	
		OPTIONS="-pin -nblocks $blocks -size $size"
		
		cd $ROOTDIR
		filename=$TIMINGDIR/sched.$STARPU_SCHED.$size

		for iter in `seq 1 $maxiter`
		do
			echo "$iter / $maxiter"
			 echo "$ROOTDIR/examples/cholesky/dw_cholesky $OPTIONS 2> /dev/null"
			 val=`$MS_LAUNCHER $STARPU_LAUNCH $ROOTDIR/examples/cholesky/dw_cholesky $OPTIONS 2> /dev/null`
			 echo "$val" >> $filename
		done
	done
}

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings-sched/
mkdir -p $TIMINGDIR

schedlist="greedy no-prio dm random"
#schedlist="random"

for sched in $schedlist
do
	export STARPU_SCHED=$sched
	echo "sched : $STARPU_SCHED"

	trace_sched;
done
