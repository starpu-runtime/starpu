#/bin/bash
DIR=$PWD
ROOTDIR=$DIR/../..

TIMINGDIR=$DIR/timings/

mkdir -p $TIMINGDIR
cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make examples ATLAS=1 CPUS=16 1> /dev/null 2> /dev/null

echo "speedup ..."

for size in 2048 4096 8192
do	
	echo "# ncpus	time	reftime" >  $TIMINGDIR/speedup.$size

	for cpus in 1 2 4 6 8 10 12 14 16
	do
		export NCPUS=$cpus

		echo "ncpus $cpus size $size"

		filename=$TIMINGDIR/timing.$cpus.$size
		$ROOTDIR/examples/mult/dw_mult -x $size -y $size -z $size -nblocks 16 2>/dev/null| tee $filename

		echo "$cpus	`cat $TIMINGDIR/timing.$cpus.$size`	`cat  $TIMINGDIR/timing.1.$size`" >> $TIMINGDIR/speedup.$size
	done
done
