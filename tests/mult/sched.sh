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
maxiter=5

# Testing another specific scheduler, no need to run this
[ -z "$STARPU_SCHED" -o "$STARPU_SCHED" = random -o "$STARPU_SCHED" = greedy -o "$STARPU_SCHED" = dm ] || exit 77

trace_sched()
{
	sched=$1

	for blocks in `seq 2 1 24`
	do
		size=$(($blocks*1024))
	
		echo "size : $size"
	
		OPTIONS="-pin -nblocks $blocks -x $size -y $size -z 1024"
		
		cd $ROOTDIR
		filename=$TIMINGDIR/sched.$sched.$size
		
		for iter in `seq 1 $maxiter`
		do
			echo "$iter / $maxiter"
			#echo "$ROOTDIR/examples/mult/sgemm $OPTIONS 2> /dev/null"
			STARPU_NCUDA=1 STARPU_CALIBRATE=1 STARPU_SCHED="$sched" $MS_LAUNCHER $STARPU_LAUNCH $ROOTDIR/examples/mult/sgemm $OPTIONS 2> /dev/null >> $filename
			tail -1 $filename
		done
	done
}

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings-sched/
mkdir -p $TIMINGDIR

schedlist="random random random greedy greedy dm"

for iter in `seq 1 10000`
do
echo "ITER $iter"
for sched in $schedlist
do
	trace_sched $sched;
done

done
