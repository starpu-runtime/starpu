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
			 val=`$ROOTDIR/examples/cholesky/dw_cholesky $OPTIONS 2> /dev/null`
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
