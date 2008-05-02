#/bin/bash
DIR=$PWD
ROOTDIR=$DIR/../..

TIMINGDIR=$DIR/timings/

mkdir -p $TIMINGDIR
cd $ROOTDIR

make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CUBLAS=1 CPUS=4 1> /dev/null 2> /dev/null

BLOCKS=8

THETALIST="32 64 128 192 256 384"

echo "absolute wall time ..."
# perform all measurments first 
for theta in $THETALIST
do
	size=$(($theta*32))

	for cpus in 0
	do
		for cublas in 1
		do
			blocks=$BLOCKS
			filename=$TIMINGDIR/timing.$cpus.$cublas.$size.$blocks

			export NCPUS=$cpus
			export NCUBLAS=$cublas

			echo "size $size cpus $cpus cublas $cublas blocks $blocks" 
			$ROOTDIR/examples/heat -nthick 34 -ntheta $(($theta+2)) -nblocks $BLOCKS 2>/dev/null| tee $filename
		done
	done


	for cpus in 1 2 3 4
	do
		for cublas in 0 1
		do
			blocks=$BLOCKS
			filename=$TIMINGDIR/timing.$cpus.$cublas.$size.$blocks

			export NCPUS=$cpus
			export NCUBLAS=$cublas

			echo "size $size cpus $cpus cublas $cublas blocks $blocks" 
			$ROOTDIR/examples/heat -nthick 34 -ntheta $(($theta+2)) -nblocks $BLOCKS 2>/dev/null| tee $filename
		done
	done
done

# time
rm -f $DIR/time
for theta in $THETALIST
do
	size=$(($theta*32))
	line=`cat  $TIMINGDIR/timing.0.1.$size.$BLOCKS $TIMINGDIR/timing.4.0.$size.$BLOCKS $TIMINGDIR/timing.1.1.$size.$BLOCKS $TIMINGDIR/timing.3.1.$size.$BLOCKS $TIMINGDIR/timing.4.1.$size.$BLOCKS | tr '\n' '\t'`
	echo "$size	$line" >> $DIR/time
done

echo "speedup ..."

for blocks in 2 4 8 16 32
do	
	for theta in $THETALIST
	do
		size=$(($theta*32))

		export NCPUS=4
		export NCUBLAS=0

		echo "size $size cpus 4 cublas 0 blocks $blocks"
		filename=$TIMINGDIR/timing.4.0.$size.$blocks
		$ROOTDIR/examples/heat -nthick 34 -ntheta $(($theta+2)) -nblocks $blocks 2>/dev/null| tee $filename

		export NCPUS=3
		export NCUBLAS=1

		echo "size $size cpus 3 cublas 1 blocks $blocks"
		filename=$TIMINGDIR/timing.3.1.$size.$blocks
		$ROOTDIR/examples/heat -nthick 34 -ntheta $(($theta+2)) -nblocks $blocks 2>/dev/null| tee $filename
	done
done

# speedups 
for blocks in 2 4 8 16 32
do
	rm -f $DIR/speedup.$blocks
	for theta in $THETALIST
	do
		size=$(($theta*32))

		echo "$size	`cat $TIMINGDIR/timing.3.1.$size.$blocks`	`cat  $TIMINGDIR/timing.4.0.$size.$blocks`" >> $DIR/speedup.$blocks 
	done
done
