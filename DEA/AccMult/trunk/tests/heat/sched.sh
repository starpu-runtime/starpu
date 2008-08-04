#!/bin/bash

maxiter=1
MAXCPU=3

trace_sched()
{
	for blocks in `seq 20 2 20`
	do
		ntheta=$(( $(($blocks*32)) + 2))
		size=$(( $(($blocks*32)) * 32))
	
		echo "size : $size"
	
		OPTIONS="-pin -v2 -nblocks $blocks -ntheta $ntheta -nthick 34"
		
		cd $ROOTDIR
		filename=$TIMINGDIR/sched.$SCHED.$size
		#rm -f $filename
		make clean 1> /dev/null 2> /dev/null
		make examples -j ATLAS=1 CPUS=$MAXCPU CUDA=1 1> /dev/null 2> /dev/null
		cd $DIR
		
		for iter in `seq 1 $maxiter`
		do
			echo "$iter / $maxiter"
			 val=`$ROOTDIR/examples/heat/heat $OPTIONS 2> /dev/null`
			 echo "$val" >> $filename
		done
	done
}

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings-sched/
mkdir -p $TIMINGDIR

#schedlist="greedy prio dm"
schedlist="random"

for sched in $schedlist
do
	export SCHED=$sched
	echo "sched : $SCHED"

	trace_sched;
done
