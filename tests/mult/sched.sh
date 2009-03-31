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


maxiter=10
MAXCPU=3

trace_sched()
{
	for blocks in `seq 2 1 24`
	do
		size=$(($blocks*1024))
	
		echo "size : $size"
	
		OPTIONS="-pin -nblocks $blocks -x $size -y $size -z $size"
		
		cd $ROOTDIR
		filename=$TIMINGDIR/sched.$SCHED.$size
		#rm -f $filename
		make clean 1> /dev/null 2> /dev/null
		make examples -j ATLAS=1 CPUS=$MAXCPU CUDA=1 1> /dev/null 2> /dev/null
		cd $DIR
		
		for iter in `seq 1 $maxiter`
		do
			echo "$iter / $maxiter"
			 echo "$ROOTDIR/examples/mult/dw_mult $OPTIONS 2> /dev/null"
			 val=`$ROOTDIR/examples/mult/dw_mult $OPTIONS 2> /dev/null`
			 echo "$val" >> $filename
		done
	done
}

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings-sched/
mkdir -p $TIMINGDIR

schedlist="random greedy model random random random greedy dm greedy dm random random random random"
#schedlist="random"

for sched in $schedlist
do
	export SCHED=$sched
	echo "sched : $SCHED"

	trace_sched;
done
