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

maxiter=5
ROOTDIR=../../
TIMINGDIR=$PWD/timing/

export WORKERS_GPUID="1"

trace_sched()
{
	sched=$1
	use_prio=$2

	export SCHED=$sched

	for blocks in `seq 24 2 24`
	do
		size=$(($blocks*1024))
	
		echo "size : $size"
	
		OPTIONS="-pin -nblocks $blocks -size $size"

		if [ $use_prio -eq 0 ]
		then
			OPTIONS="$OPTIONS -no-prio"
		fi
		
		filename=$TIMINGDIR/sched.$SCHED.$size.$use_prio

		for iter in `seq 1 $maxiter`
		do
			echo "$iter / $maxiter"
			echo "$ROOTDIR/examples/cholesky/dw_cholesky $OPTIONS 2> /dev/null"
			val=`$ROOTDIR/examples/cholesky/dw_cholesky $OPTIONS 2> /dev/null`
			echo "$val" >> $filename
			echo "$val"
		done
	done
}

schedlist='dm dm dm dm greedy dm'

export NCUDA=1
export CALIBRATE=1

mkdir -p $TIMINGDIR

# calibrate
for i in `seq 1 5` 
do
SCHED="dm" $ROOTDIR/examples/cholesky/dw_cholesky -nblocks 16 -size 16384 2> /dev/null
done

for sched in $schedlist
do
	echo "sched : $sched"

	trace_sched $sched 0;
	trace_sched $sched 1;
done
