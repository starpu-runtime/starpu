#!/bin/bash

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings/
mkdir -p $TIMINGDIR
filename=$TIMINGDIR/gflops.data

tilelist="256 512 1024 2048"
sizelist="1024 2048 4096 8192 16384"

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

		if [ $tile -lt $size -a $nblocks -lt 32 -a $(($size % $tile)) == 0 ];
		then
			echo "start tile $tile size $size nblocks $nblocks  "
			timing=`$ROOTDIR/examples/dw_mult -pin -x $size -y $size -z $size -nblocks $nblocks 2>/dev/null`
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
make ATLAS=1 CUBLAS=1 CPUS=3 1> /dev/null 2> /dev/null

cd $DIR

trace_header 
for size in $sizelist
do
	trace_size $size;
done
