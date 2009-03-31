#!/bin/bash

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


maxiter=1
MAXCPU=3

MINSIZE=$((17*1024))
MAXSIZE=$((29*1024))

trace_granularity()
{
	grain=$1

	#minblocks=1
	minblocks=$(($MINSIZE/$grain))
	#maxblocks=2
	maxblocks=$(($MAXSIZE/$grain))

	#step=2
	step=2

	for blocks in `seq $minblocks $step $maxblocks`
	do
		size=$(($blocks*$grain))
		
		ntheta=$(( $(($size/32)) + 2))
	
		echo "size : $size (grain $grain nblocks $blocks)"
	
		OPTIONS="-pin -nblocks $blocks -ntheta $ntheta -nthick 34 -v2"
		
		filename=$TIMINGDIR/granularity.$grain.$size
		#rm -f $filename
		
		for iter in `seq 1 $maxiter`
		do
			echo "$iter / $maxiter"
			 val=`SCHED="dm" $ROOTDIR/examples/heat/heat $OPTIONS 2> /dev/null`
			 echo "$val" >> $filename
		done
	done
}


trace_granularity_nomodel()
{
	grain=$1

	#minblocks=1
	minblocks=$(($MINSIZE/$grain))
	#maxblocks=2
	maxblocks=$(($MAXSIZE/$grain))

	step=2

	for blocks in `seq $minblocks $step $maxblocks`
	do
		size=$(($blocks*$grain))
		
		ntheta=$(( $(($size/32)) + 2))
	
		echo "size : $size (grain $grain nblocks $blocks)"
	
		OPTIONS="-pin -nblocks $blocks -ntheta $ntheta -nthick 34 -v2"
		
		filename=$TIMINGDIR/granularity.nomodel.$grain.$size
		#rm -f $filename
		
		for iter in `seq 1 $maxiter`
		do
			echo "$iter / $maxiter"
			 val=`SCHED="greedy" $ROOTDIR/examples/heat/heat $OPTIONS 2> /dev/null`
			 echo "$val" >> $filename
		done
	done
}



calibrate_grain()
{
	grain=$1;


	# calibrate with 12k problems
	blocks=$((12288/$grain))
	ntheta=$((384+2))

#	#in case this is *really* a small granularity, only 4K
#	blocks=$((4096/$grain))
#	ntheta=$((128+2))
#
#	blocks=$((2048/$grain))
#	ntheta=$((64+2))
#
#	blocks=8
#	ntheta=$((2+$(($size/32))))

	size=$(($blocks*$grain))
	echo "Calibrating grain $grain size $size ($blocks blocks)"

	for iter in `seq 1 4`
	do
		OPTIONS="-pin -nblocks $blocks -ntheta $ntheta -nthick 34 -v2"

		val=`CALIBRATE=1 SCHED="dm" $ROOTDIR/examples/heat/heat $OPTIONS `
	done
	
}

DIR=$PWD
ROOTDIR=$DIR/../..
SAMPLINGDIR=$DIR/sampling/
TIMINGDIR=$DIR/timing/
mkdir -p $TIMINGDIR
mkdir -p $SAMPLINGDIR
#rm  -f $SAMPLINGDIR/*

#grainlist="64 128 256 512 768 1024 1536 2048"
grainlist="1024 512 256"
#grainlist="1280"

export PERF_MODEL_DIR=$SAMPLINGDIR

cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make examples -j ATLAS=1 CPUS=$MAXCPU CUDA=1 1> /dev/null 2> /dev/null

cd $DIR

# calibrate (sampling)
#for grain in $grainlist
#do
#	calibrate_grain $grain;
#done

# perform the actual benchmarking now
for grain in $grainlist
do
	trace_granularity $grain;	
#	trace_granularity_nomodel $grain;
done
