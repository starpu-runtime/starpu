#!/bin/bash

#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#


DIR=$PWD
ROOTDIR=$DIR/../..
COVDIR=coverage
MAXCPU=3

init()
{
	mkdir -p $COVDIR
	lcov --directory $COVDIR --zerocounters > /dev/null
}

save_cov()
{
	testname=$1
	lcov --directory $ROOTDIR --capture --output $COVDIR/$testname.info > /dev/null 
	lcov -a $COVDIR/$testname.info -o $COVDIR/all.info > /dev/null
}

generatehtml()
{
	cd $COVDIR
	genhtml all.info
	cd $DIR
}

apps()
{

echo "incrementer"
timing=`$ROOTDIR/examples/incrementer/incrementer 2> /dev/null`
save_cov "incrementer";

echo "tag_example"
timing=`$ROOTDIR/examples/tag_example/tag_example -iter 64 -i 128 -j 24 2> /dev/null`
save_cov "tag_example";

echo "tag_example2"
timing=`$ROOTDIR/examples/tag_example/tag_example2 -iter 64 -i 128 2> /dev/null`
save_cov "tag_example2";

echo "spmv"
timing=`$ROOTDIR/examples/spmv/dw_spmv 2> /dev/null`
save_cov "spmv";

echo "spmv.gpu"
timing=`NCPUS=0 $ROOTDIR/examples/spmv/dw_spmv 2> /dev/null`
save_cov "spmv.gpu";

echo "spmv.cpu"
timing=`NCUDA=0 $ROOTDIR/examples/spmv/dw_spmv 2> /dev/null`
save_cov "spmv.cpu";

echo "spmv.dm"
timing=`SCHED="dm" $ROOTDIR/examples/spmv/dw_spmv 2> /dev/null`
save_cov "spmv.dm";

echo "spmv.dmda"
timing=`SCHED="dmda" $ROOTDIR/examples/spmv/dw_spmv 2> /dev/null`
save_cov "spmv.dmda";


echo "strassen.ws"
timing=`SCHED="ws" $ROOTDIR/examples/strassen/dw_strassen -rec 3 -size 2048 -pin 2> /dev/null`
save_cov "strassen.ws";


echo "strassen.dm"
timing=`SCHED="dm" $ROOTDIR/examples/strassen/dw_strassen -rec 3 -size 2048 -pin 2> /dev/null`
save_cov "strassen.dm";


echo "strassen.dmda"
timing=`SCHED="dmda" $ROOTDIR/examples/strassen/dw_strassen -rec 3 -size 2048 -pin 2> /dev/null`
save_cov "strassen.dmda";

echo "chol.dm"
timing=`CALIBRATE=1 SCHED="dm" $ROOTDIR/examples/cholesky/dw_cholesky -pin 2> /dev/null`
save_cov "chol.dm";


echo "chol.dmda"
timing=`CALIBRATE=1 SCHED="dmda" $ROOTDIR/examples/cholesky/dw_cholesky -pin 2> /dev/null`
save_cov "chol.dmda";

echo "chol.cpu"
timing=`CALIBRATE=1 NCUDA=0 SCHED="dm" $ROOTDIR/examples/cholesky/dw_cholesky -pin 2> /dev/null`
save_cov "chol.cpu";

echo "chol.gpu"
timing=`CALIBRATE=1 NCPUS=0 SCHED="dm" $ROOTDIR/examples/cholesky/dw_cholesky -pin 2> /dev/null`
save_cov "chol.gpu";

echo "chol"
timing=`$ROOTDIR/examples/cholesky/dw_cholesky 2> /dev/null`
save_cov "chol";

echo "heat.dm.4k.calibrate.v2"
timing=`CALIBRATE=1 SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 66 -nblocks 4 -v2 -pin 2> /dev/null`
save_cov "heat.dm.4k.calibrate.v2";


echo "heat.dm.8k.calibrate.v2"
timing=`CALIBRATE=1 SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin 2> /dev/null`
save_cov "heat.dm.8k.calibrate.v2";


echo "heat.dm.16k.calibrate.v2"
timing=`CALIBRATE=1 SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 130 -nthick 130 -nblocks 16 -v2 -pin 2> /dev/null`
save_cov "heat.dm.16k.calibrate.v2";



echo "heat.dm.8k.no.pin.v2"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 2> /dev/null`
save_cov "heat.dm.8k.no.pin.v2";

echo "heat.prio.8k"
timing=`SCHED="prio" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin 2> /dev/null`
save_cov "heat.prio.8k";

echo "heat.dm.8k.v2.no.prio"
timing=`SCHED="no-prio" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.dm.8k.v2.no.prio";

echo "heat.dm.8k.v2.random"
timing=`SCHED="random" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.dm.8k.v2.random";

echo "heat.dm.8k.v2"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.dm.8k.v2";

echo "heat.dm.16k.v2"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 130 -nthick 130 -nblocks 16 -pin -v2 2> /dev/null`
save_cov "heat.dm.16k.v2";

echo "heat.ws.8k.v2"
timing=`SCHED="ws" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.ws.8k.v2";

echo "heat.greedy.8k.v2"
timing=`SCHED="greedy" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.greedy.8k.v2";

echo "heat.8k.cg"
timing=`$ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg 2> /dev/null`
save_cov "heat.8k.cg";


echo "heat.dm.8k.cg"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg 2> /dev/null`
save_cov "heat.dm.8k.cg";

echo "heat.dm.8k.v3"
timing=`SCHED="dm" $ROOTDIR/examples/heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v3 2> /dev/null`
save_cov "heat.dm.8k.v3";

echo "mult.dm.common"
timing=`SCHED="dm" $ROOTDIR/examples/mult/dw_mult -nblocks 4 -x 4096 -y 4096 -z 1024 -pin -common-model 2> /dev/null`
save_cov "mult.dm.common";

echo "mult.dm"
timing=`CALIBRATE=1 SCHED="dm" $ROOTDIR/examples/mult/dw_mult -nblocks 8 -x 8192 -y 8192 -z 8192 -pin 2> /dev/null`
save_cov "mult.dm";

echo "mult.dmda"
timing=`CALIBRATE=1 SCHED="dmda" $ROOTDIR/examples/mult/dw_mult -nblocks 8 -x 8192 -y 8192 -z 8192 -pin 2> /dev/null`
save_cov "mult.dmda";


}

cd $ROOTDIR
./configure --enable-coverage > /dev/null

make clean 1> /dev/null 2> /dev/null
make examples -j 1> /dev/null 2> log
cd $DIR

init;

apps;

cd $ROOTDIR
make clean 1> /dev/null 2> /dev/null
make examples -j NO_DATA_RW_LOCK=1 1> /dev/null 2> log
cd $DIR


apps;

generatehtml;
