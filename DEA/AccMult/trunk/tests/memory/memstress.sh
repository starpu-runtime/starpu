#!/bin/bash

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings/
mkdir -p $TIMINGDIR
filename=$TIMINGDIR/memstress.data

sizelist="4096 8192"
stresslist="0 50 100 150 200 250 300 350 400 450 500 550 600 650 655 660 665 670 675"
#stresslist="672"

trace_stress()
{
	memstress=$1

	export NCPUS=0
	export NCUBLAS=1
	export STRESS_MEM=$memstress

	line="$memstress"

	for size in $sizelist 
	do
		nblocks=$(($size / 1024))
		echo "Computing size $size with $memstress MB of memory LESS"
		
		echo "$ROOTDIR/examples/dw_mult -x $size -y $size -z $size -nblocks $nblocks 2>/dev/null"
		timing=`$ROOTDIR/examples/dw_mult -x $size -y $size -z $size -nblocks $nblocks 2>/dev/null`
	
		echo "size : $size memstress $memstress => $timing us"

		line="$line	$timing"

	done

	echo "$line" >> $filename
}

cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CUBLAS=1 CPUS=3 1> /dev/null 2> /dev/null

cd $DIR

echo "#memstress $sizelist " > $filename

for memstress in $stresslist
do
	trace_stress $memstress;
done
