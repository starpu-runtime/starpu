#!/bin/bash

maxiter=100

DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings/
mkdir -p $TIMINGDIR

cd $ROOTDIR
filename=$TIMINGDIR/sched.greedy.data
rm -f $filename
make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CPUS=4 1> /dev/null 2> /dev/null
cd $DIR

for iter in `seq 1 $maxiter`
do
	echo "$iter / $maxiter"
	 val=`$ROOTDIR/examples/heat 2> /dev/null`
	 echo "$val" >> $filename
done



cd $ROOTDIR
filename=$TIMINGDIR/sched.greedy.noprio.data
rm -f $filename
make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CPUS=4 NO_PRIO=1 1> /dev/null 2> /dev/null
cd $DIR

for iter in `seq 1 $maxiter`
do
	echo "$iter / $maxiter"
	 val=`$ROOTDIR/examples/heat 2> /dev/null`
	 echo "$val" >> $filename
done

cd $ROOTDIR
filename=$TIMINGDIR/sched.greedy.ws.data
rm -f $filename
make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CPUS=4 1> /dev/null 2> /dev/null
cd $DIR

for iter in `seq 1 $maxiter`
do
	echo "$iter / $maxiter"
	 val=`SCHED=ws $ROOTDIR/examples/heat 2> /dev/null`
	 echo "$val" >> $filename
done



cd $ROOTDIR
filename=$TIMINGDIR/sched.greedy.noprio.ws.data
rm -f $filename
make clean 1> /dev/null 2> /dev/null
make ATLAS=1 CPUS=4 NO_PRIO=1 1> /dev/null 2> /dev/null
cd $DIR

for iter in `seq 1 $maxiter`
do
	echo "$iter / $maxiter"
	 val=`SCHED=ws $ROOTDIR/examples/heat 2> /dev/null`
	 echo "$val" >> $filename
done
