#!/bin/bash

maxiter=1
MAXCPU=3

MINSIZE=$((1*1024))
MAXSIZE=$((30*1024))

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

	#step=2
	step=1

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
	blocks=8
	size=$((8*$grain))
	ntheta=$((2+$(($size/32))))

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
grainlist="768 1024 512"
#grainlist="1280"

export PERF_MODEL_DIR=$SAMPLINGDIR

cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make examples -j ATLAS=1 CPUS=$MAXCPU CUDA=1 1> /dev/null 2> /dev/null

cd $DIR
#
## calibrate (sampling)
#for grain in $grainlist
#do
#	calibrate_grain $grain;
#done

# perform the actual benchmarking now
for grain in $grainlist
do
#	trace_granularity $grain;	
	trace_granularity_nomodel $grain;
done
