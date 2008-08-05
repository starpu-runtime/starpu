#!/bin/bash

ampllist="0.0 0.1 0.2 0.25 0.3 0.4 0.50 0.6 0.7 0.75 0.8 0.9 0.95 1.0 "
#ampllist="0.50 0.75 0.95 1.0 "

maxiter=10
MAXCPU=3

trace_perturbation()
{
	export SCHED="dm"

	for blocks in `seq 2 2 16`
	do

		ntheta=$(( $(($blocks*32)) + 2))
		size=$(( $(($blocks*32)) * 32))
	
		echo "size : $size"
	
		OPTIONS="-pin -v2 -nblocks $blocks -ntheta $ntheta -nthick 34"
		
		cd $ROOTDIR
		filename=$TIMINGDIR/pertubate.$size.$AMPL
		#rm -f $filename
		make clean 1> /dev/null 2> /dev/null
		make examples -j ATLAS=1 CPUS=$MAXCPU CUDA=1 PERTURB_AMPL=$AMPL 1> /dev/null 2> /dev/null
		cd $DIR

		if [ $size -le 16384 ]
		then
			nsamples=$maxiter
		else
			nsamples=2
		fi
		
		for iter in `seq 1 $nsamples`
		do
			echo "$iter / $nsamples"
			 val=`$ROOTDIR/examples/heat/heat $OPTIONS 2> /dev/null`
			 echo "$val" >> $filename
		done
	done
}

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timing-perturbate/
mkdir -p $TIMINGDIR

for ampl in $ampllist
do
	export AMPL=$ampl
	echo "ampl : $AMPL"

	trace_perturbation;
done
