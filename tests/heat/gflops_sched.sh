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

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings/
mkdir -p $TIMINGDIR

#tilelist="128 256 512 1024"
#sizelist="256 512 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600"

tilelist="1024"
sizelist="1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 16384 20480 22528 24576 25600"
#sizelist="18432 20480 22528 24576 25600"
# 16384 20480 24576 25600"

heat_ret=0
policy=none

measure_heat()
{
	thick=$1
	theta=$2
	nblocks=$3
	size=$4

	if [ $size -le 16384 ] 
	then
		nsample=1
	else
		nsample=1
	fi

	total=0

	for i in `seq 1 $nsample`
	do
		echo "iter $i/$nsample"
		val=`STARPU_SCHED=$policy $MS_LAUNCHER $STARPU_LAUNCH $ROOTDIR/examples/heat/heat -nthick $thick -ntheta $theta -nblocks $nblocks -pin -v2 2>/dev/null`
		total=`echo "$val + $total" |bc -l`
	done

	heat_ret=`echo "$total / $nsample"|bc -l`
}

trace_header()
{
	line="# size 	"
	for tile in $tilelist
	do
		line="$line	$tile"
	done

	echo "$line" > $filename
}

trace_size()
{
	size=$1

	echo "Computing size $size"
	
	line="$size"

	for tile in $tilelist
	do
		nblocks=$(($size / $tile))

		theta=$(($(($size / 32)) + 2))
		thick=34

		if [ $tile -le $size -a $nblocks -le 32 -a $(($size % $tile)) == 0 ];
		then
			echo "STARPU_SCHED=$policy $ROOTDIR/examples/heat/heat -nthick $thick -ntheta $theta -nblocks $nblocks -pin -v2"
			measure_heat $thick $theta $nblocks $size;
			timing=$heat_ret
		else
			timing="x"
		fi

		echo "size : $size tile $tile => $timing us"

		line="$line	$timing"

	done

	echo "$line" >> $filename
}

cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make examples STARPU_ATLAS=1 CUDA=1 CPUS=3 1> /dev/null 2> log

cd $DIR

filename=$TIMINGDIR/gflops.greedy.data
policy=greedy
trace_header 
for size in $sizelist
do
	trace_size $size;
done

cd $DIR

filename=$TIMINGDIR/gflops.prio.data
policy=prio
trace_header 
for size in $sizelist
do
	trace_size $size;
done

filename=$TIMINGDIR/gflops.ws.data
policy=ws
trace_header 
for size in $sizelist
do
	trace_size $size;
done


filename=$TIMINGDIR/gflops.lws.data
policy=lws
trace_header 
for size in $sizelist
do
	trace_size $size;
done


filename=$TIMINGDIR/gflops.noprio.data
policy=no-prio
trace_header 
for size in $sizelist
do
	trace_size $size;
done



paste $TIMINGDIR/gflops.greedy.data $TIMINGDIR/gflops.prio.data $TIMINGDIR/gflops.ws.data $TIMINGDIR/gflops.noprio.data > $TIMINGDIR/gflops.merged.data
