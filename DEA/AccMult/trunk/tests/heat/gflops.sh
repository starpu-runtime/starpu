#!/bin/bash

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings/
mkdir -p $TIMINGDIR
filename=$TIMINGDIR/gflops.data

tilelist="128 256 512 1024 2048"
sizelist="256 512 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600"

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

		if [ $tile -lt $size -a $nblocks -lt 32 -a $(($size % $tile)) == 0 ];
		then
			timing=`$ROOTDIR/examples/heat -nthick $thick -ntheta $theta -nblocks $nblocks 2>/dev/null`
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
make ATLAS=1 CUBLAS=1 CPUS=3 1> /dev/null 2> /dev/null

cd $DIR

trace_header 
for size in $sizelist
do
	trace_size $size;
done
