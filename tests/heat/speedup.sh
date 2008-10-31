#/bin/bash
DIR=$PWD
ROOTDIR=$DIR/../..

TIMINGDIR=$DIR/timings/

mkdir -p $TIMINGDIR
cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CPUS=16 1> /dev/null 2> /dev/null

echo "speedup ..."

for theta in 32 64 128 256 512 1024
do	
	size=$(($theta * 32))

	echo "# ncpus	time	reftime" >  $TIMINGDIR/speedup.$size

	for cpus in 1 2 4 6 8 10 12 14 16
	do
		export NCPUS=$cpus

		echo "ncpus $cpus size $size"

		filename=$TIMINGDIR/timing.$cpus.$size
		$ROOTDIR/examples/heat -v2 -pin -nthick 34 -ntheta $(($theta+2)) -nblocks 16 2>/dev/null| tee $filename

		echo "$cpus	`cat $TIMINGDIR/timing.$cpus.$size`	`cat  $TIMINGDIR/timing.1.$size`" >> $TIMINGDIR/speedup.$size
	done
done
