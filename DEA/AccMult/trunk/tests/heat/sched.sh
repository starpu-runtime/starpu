#!/bin/bash

maxiter=10
MAXCPU=3

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings-sched/
mkdir -p $TIMINGDIR

for blocks in `seq 2 2 16`
do
	ntheta=$(( $(($blocks*32)) + 2))
	size=$(( $(($blocks*32)) * 32))

	echo "size : $size"

	OPTIONS="-pin -v3 -nblocks $blocks -ntheta $ntheta -nthick 34"
	
	cd $ROOTDIR
	filename=$TIMINGDIR/sched.greedy.$size
	rm -f $filename
	make clean 1> /dev/null 2> /dev/null
	make ATLAS=1 CPUS=$MAXCPU CUDA=1 1> /dev/null 2> /dev/null
	cd $DIR
	
	for iter in `seq 1 $maxiter`
	do
		echo "$iter / $maxiter"
		 val=`$ROOTDIR/examples/heat $OPTIONS 2> /dev/null`
		 echo "$val" >> $filename
	done
	
	cd $ROOTDIR
	filename=$TIMINGDIR/sched.ws.$size
	rm -f $filename
	make clean 1> /dev/null 2> /dev/null
	make ATLAS=1 CPUS=$MAXCPU CUDA=1 1> /dev/null 2> /dev/null
	cd $DIR
	
	for iter in `seq 1 $maxiter`
	do
		echo "$iter / $maxiter"
		 val=`SCHED=ws $ROOTDIR/examples/heat $OPTIONS 2> /dev/null`
		 echo "$val" >> $filename
	done
	
	
	cd $ROOTDIR
	filename=$TIMINGDIR/sched.ws.overload.$size
	rm -f $filename
	make clean 1> /dev/null 2> /dev/null
	make ATLAS=1 CPUS=$MAXCPU CUDA=1 1> /dev/null 2> /dev/null
	cd $DIR
	
	for iter in `seq 1 $maxiter`
	do
		echo "$iter / $maxiter"
		 val=`SCHED=ws $ROOTDIR/examples/heat $OPTIONS USE_OVERLOAD=1 2> /dev/null`
		 echo "$val" >> $filename
	done
	
done
