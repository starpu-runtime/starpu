#/bin/bash
DIR=$PWD
ROOTDIR=$DIR/../..

TIMINGDIR=$DIR/timings/

mkdir -p $TIMINGDIR
cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CUBLAS=1 CPUS=4 1> /dev/null 2> /dev/null

BLOCKS=8

for theta in 32 64 128 256
do
	size=$(($theta*32))
	for cpus in 1 2 3 4
	do
		for cublas in 0 1
		do
			blocks=$BLOCKS
			filename=$TIMINGDIR/timing.$cpus.$cublas.$size.$blocks

			NCPUS=$cpus
			NCUBLAS=$cublas

			echo "size $size cpus $cpus cublas $cublas blocks $blocks" 
			$ROOTDIR/examples/heat -nthick 34 -ntheta $(($theta+2)) -nblocks $BLOCKS 2>/dev/null| tee $filename
		done
	done
done


