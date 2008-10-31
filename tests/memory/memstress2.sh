#!/bin/bash

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings/
mkdir -p $TIMINGDIR
filename=$TIMINGDIR/memstress2.data

sizelist="512 1024 2048 4096 8192 16384"
stresslist="0 350"
#stresslist="672"

trace_stress()
{
	size=$1

	line="$size"

	for stress in $stresslist
	do
		export STRESS_MEM=$stress

		nblocks=$(($size / 1024))
		echo "Computing size $size with $stress MB of memory LESS"

		
		echo "$ROOTDIR/examples/mult/dw_mult -x $size -y $size -z $size -nblocks $nblocks 2>/dev/null"
		timing=`$ROOTDIR/examples/mult/dw_mult -x $size -y $size -z $size -nblocks $nblocks 2>/dev/null`
	
		echo "size : $size memstress $stress => $timing us"

		line="$line	$timing"

	done

	echo "$line" >> $filename
}

cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make examples ATLAS=1 CUDA=1 CPUS=0 1> /dev/null 2> /dev/null

cd $DIR

echo "#size $stresslist " > $filename

for size in $sizelist
do
	trace_stress $size;
done
