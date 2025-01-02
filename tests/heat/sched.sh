#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2008-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
[ -z "$STARPU_SCHED" -o "$STARPU_SCHED" = greedy -o "$STARPU_SCHED" = dm -o "$STARPU_SCHED" = dmda ] || exit 77

calibrate_point()
{
	strat=$1
	nblocks=$2
	prefetch=$3

	size=$(($nblocks * 1024))

	echo "STARPU_CALIBRATE size : $size / blocks : $nblocks strat -> $strat prefetch -> $prefetch"

	rm -f $SAMPLINGDIR/*

	for iter in `seq 1 $maxiter`
	do
		echo "$iter / $maxiter"
		export STARPU_SCHED=$strat
		export STARPU_CALIBRATE=1
		export STARPU_PREFETCH=$prefetch
		val=`$MS_LAUNCHER $STARPU_LAUNCH $ROOTDIR/examples/heat/heat -pin -nblocks $nblocks -size $size -v3 2> /dev/null`
		echo "$val"
	done

}

trace_point()
{
	strat=$1
	nblocks=$2
	docalibrate=$3
	prefetch=$4

	size=$(($nblocks * 1024))

	echo "size : $size ... docalibrate $docalibrate"

	filename=$TIMINGDIR/sched.$prefetch.$strat.$size

	if [ $docalibrate == 1 ];
	then
		calibrate_point $strat $nblocks $prefetch
	fi

	for iter in `seq 1 $maxiter`
	do
		echo "$iter / $maxiter"
		export STARPU_SCHED=$strat
		export STARPU_CALIBRATE=$docalibrate
		export STARPU_PREFETCH=$prefetch
		val=`$MS_LAUNCHER $STARPU_LAUNCH $ROOTDIR/examples/heat/heat -pin -nblocks $nblocks -size $size -v3  2> /dev/null`
		echo "$val"
		echo "$val" >> $filename
	done
}

trace_sched()
{
	strat=$1
	prefetch=$2
	docalibrate=$3

	for nblocks in `seq 2 2 30`
	do
		trace_point $strat $nblocks $docalibrate $prefetch
	done
}


DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings_sched/
SAMPLINGDIR=$DIR/sampling/
export STARPU_PERF_MODEL_DIR=$SAMPLINGDIR

mkdir -p $TIMINGDIR
mkdir -p $SAMPLINGDIR

#schedlist="ws lws no-prio greedy prio dm random"
#schedlist="random random random random"

export STARPU_NCUDA=3
export STARPU_NCPUS=8

trace_sched "greedy" 0 0;
trace_sched "dm" 0 1;
trace_sched "dm" 1 1;
trace_sched "dmda" 1 1;
