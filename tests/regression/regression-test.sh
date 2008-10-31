#!/bin/bash

DIR=$PWD
ROOTDIR=$DIR/../..
COVDIR=coverage
MAXCPU=3

init()
{
	mkdir -p $COVDIR
	lcov --directory $COVDIR --zerocounters
}

save_cov()
{
	testname=$1
	lcov --directory $ROOTDIR --capture --output $COVDIR/$testname.info 
	lcov -a $COVDIR/$testname.info -o $COVDIR/all.info
}

generatehtml()
{
	cd $COVDIR
	genhtml all.info
	cd $DIR
}

cd $ROOTDIR
make clean 1> /dev/null 2> /dev/null
make examples -j ATLAS=1 CPUS=$MAXCPU CUDA=1 COVERAGE=1 1> /dev/null 2> /dev/null
cd $DIR

init;

echo "heat.dm.8k.no.pin.v2"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 2> log`
save_cov "heat.dm.8k.no.pin.v2";

echo "heat.prio.8k"
timing=`SCHED="prio" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin 2> log`
save_cov "heat.prio.8k";

echo "heat.dm.8k.v2"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> log`
save_cov "heat.dm.8k.v2";

echo "heat.ws.8k.v2"
timing=`SCHED="ws" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> log`
save_cov "heat.ws.8k.v2";

echo "heat.greedy.8k.v2"
timing=`SCHED="greedy" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> log`
save_cov "heat.greedy.8k.v2";

echo "heat.dm.8k.cg"
timing=`$ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg 2> log`
save_cov "heat.dm.8k.cg";

echo "heat.dm.8k.v3"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v3 2> log`
save_cov "heat.dm.8k.v3";

echo "incrementer"
timing=`$ROOTDIR/examples/incrementer/incrementer 2> log`
save_cov "incrementer";

generatehtml;
