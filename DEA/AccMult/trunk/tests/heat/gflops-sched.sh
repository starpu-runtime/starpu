#!/bin/bash

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings/
mkdir -p $TIMINGDIR

#tilelist="128 256 512 1024"
#sizelist="256 512 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600"

tilelist="1024"
sizelist="1024 2048 4096 8192 16384 20480 24576 25600"
# 16384 20480 24576 25600"

heat_ret=0
policy=none

measure_heat()
{
	thick=$1
	theta=$2
	nblocks=$3
	size=$4

	if [ $size -le 2048 ] 
	then
		nsample=10
	else
		nsample=4
	fi

	total=0

	for i in `seq 1 $nsample`
	do
		echo "iter $i/$nsample"
		val=`SCHED=$policy $ROOTDIR/examples/heat -nthick $thick -ntheta $theta -nblocks $nblocks -pin -v2 2>/dev/null`
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
			echo "SCHED=$policy $ROOTDIR/examples/heat -nthick $thick -ntheta $theta -nblocks $nblocks -pin -v2"
			measure_heat $thick $theta $nblocks $size;
			timing=$heat_ret
		else
			timing="x"
		fi
	#	timing=`$ROOTDIR/examples/heat -nthick $thick -ntheta $theta -nblocks $nblocks 2>/dev/null`

		echo "size : $size tile $tile => $timing us"

		line="$line	$timing"

	done

	echo "$line" >> $filename
}

cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CUDA=1 CPUS=3 1> /dev/null 2> log

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

paste $TIMINGDIR/gflops.greedy.data $TIMINGDIR/gflops.prio.data > $TIMINGDIR/gflops.merged.data
