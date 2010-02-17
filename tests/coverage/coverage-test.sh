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
COVDIR=$PWD/coverage
BUILDDIR=$PWD/build/
INSTALLDIR=$PWD/local/
EXAMPLEDIR=$INSTALLDIR/lib/starpu/examples/

mkdir -p $INSTALLDIR
mkdir -p $BUILDDIR

init()
{
	mkdir -p $COVDIR
	lcov --directory $BUILDDIR --zerocounters > /dev/null
}

save_cov()
{
	testname=$1
	lcov --directory $BUILDDIR --capture --output $COVDIR/$testname.info > /dev/null 
	lcov -a $COVDIR/$testname.info -o $COVDIR/all.info > /dev/null
}

generatehtml()
{
	cd $COVDIR
	genhtml all.info
	cd -
}

apps()
{

echo "incrementer"
timing=`$EXAMPLEDIR/incrementer 2> /dev/null`
save_cov "incrementer";

echo "tag_example"
timing=`$EXAMPLEDIR/tag_example -iter 64 -i 128 -j 24 2> /dev/null`
save_cov "tag_example";

echo "tag_example2"
timing=`$EXAMPLEDIR/tag_example2 -iter 64 -i 128 2> /dev/null`
save_cov "tag_example2";

# echo "spmv"
# timing=`$BUILDDIR/examples/spmv/dw_spmv 2> /dev/null`
# save_cov "spmv";
# 
# echo "spmv.gpu"
# timing=`STARPU_NCPUS=0 $BUILDDIR/examples/spmv/dw_spmv 2> /dev/null`
# save_cov "spmv.gpu";
# 
# echo "spmv.cpu"
# timing=`STARPU_NCUDA=0 $BUILDDIR/examples/spmv/dw_spmv 2> /dev/null`
# save_cov "spmv.cpu";
# 
# echo "spmv.dm"
# timing=`SCHED="dm" $BUILDDIR/examples/spmv/dw_spmv 2> /dev/null`
# save_cov "spmv.dm";
# 
# echo "spmv.dmda"
# timing=`SCHED="dmda" $BUILDDIR/examples/spmv/dw_spmv 2> /dev/null`
# save_cov "spmv.dmda";

echo "strassen.ws"
timing=`SCHED="ws" $EXAMPLEDIR/dw_strassen -rec 3 -size 2048 -pin 2> /dev/null`
save_cov "strassen.ws";


echo "strassen.dm"
timing=`SCHED="dm" $EXAMPLEDIR/dw_strassen -rec 3 -size 2048 -pin 2> /dev/null`
save_cov "strassen.dm";


echo "strassen.dmda"
timing=`SCHED="dmda" $EXAMPLEDIR/dw_strassen -rec 3 -size 2048 -pin 2> /dev/null`
save_cov "strassen.dmda";

echo "chol.dm"
timing=`CALIBRATE=1 SCHED="dm" $EXAMPLEDIR/dw_cholesky -pin 2> /dev/null`
save_cov "chol.dm";


echo "chol.dmda"
timing=`CALIBRATE=1 SCHED="dmda" $EXAMPLEDIR/dw_cholesky -pin 2> /dev/null`
save_cov "chol.dmda";

echo "chol.cpu"
timing=`CALIBRATE=1 STARPU_NCUDA=0 SCHED="dm" $EXAMPLEDIR/dw_cholesky -pin 2> /dev/null`
save_cov "chol.cpu";

echo "chol.gpu"
timing=`CALIBRATE=1 STARPU_NCPUS=0 SCHED="dm" $EXAMPLEDIR/dw_cholesky -pin 2> /dev/null`
save_cov "chol.gpu";

echo "chol"
timing=`$EXAMPLE/dw_cholesky 2> /dev/null`
save_cov "chol";

echo "heat.dm.4k.calibrate.v2"
timing=`CALIBRATE=1 SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 66 -nblocks 4 -v2 -pin 2> /dev/null`
save_cov "heat.dm.4k.calibrate.v2";


echo "heat.dm.8k.calibrate.v2"
timing=`CALIBRATE=1 SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin 2> /dev/null`
save_cov "heat.dm.8k.calibrate.v2";

echo "heat.dm.16k.calibrate.v2"
timing=`CALIBRATE=1 SCHED="dm" $EXAMPLEDIR/heat -ntheta 130 -nthick 130 -nblocks 16 -v2 -pin 2> /dev/null`
save_cov "heat.dm.16k.calibrate.v2";

echo "heat.dm.8k.no.pin.v2"
timing=`SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 2> /dev/null`
save_cov "heat.dm.8k.no.pin.v2";

echo "heat.prio.8k"
timing=`SCHED="prio" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin 2> /dev/null`
save_cov "heat.prio.8k";

echo "heat.dm.8k.v2.no.prio"
timing=`SCHED="no-prio" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.dm.8k.v2.no.prio";

echo "heat.dm.8k.v2.random"
timing=`SCHED="random" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.dm.8k.v2.random";

echo "heat.dm.8k.v2"
timing=`SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.dm.8k.v2";

echo "heat.dm.16k.v2"
timing=`SCHED="dm" $EXAMPLEDIR/heat -ntheta 130 -nthick 130 -nblocks 16 -pin -v2 2> /dev/null`
save_cov "heat.dm.16k.v2";

echo "heat.ws.8k.v2"
timing=`SCHED="ws" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.ws.8k.v2";

echo "heat.greedy.8k.v2"
timing=`SCHED="greedy" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 2> /dev/null`
save_cov "heat.greedy.8k.v2";

echo "heat.8k.cg"
timing=`$EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg 2> /dev/null`
save_cov "heat.8k.cg";


echo "heat.dm.8k.cg"
timing=`SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg 2> /dev/null`
save_cov "heat.dm.8k.cg";

echo "heat.dm.8k.v3"
timing=`SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v3 2> /dev/null`
save_cov "heat.dm.8k.v3";

echo "mult.dm.common"
timing=`SCHED="dm" $EXAMPLEDIR/dw_mult -nblocks 4 -x 4096 -y 4096 -z 1024 -pin -common-model 2> /dev/null`
save_cov "mult.dm.common";

echo "mult.dm"
timing=`CALIBRATE=1 SCHED="dm" $EXAMPLEDIR/dw_mult -nblocks 8 -x 8192 -y 8192 -z 8192 -pin 2> /dev/null`
save_cov "mult.dm";

echo "mult.dmda"
timing=`CALIBRATE=1 SCHED="dmda" $EXAMPLEDIR/dw_mult -nblocks 8 -x 8192 -y 8192 -z 8192 -pin 2> /dev/null`
save_cov "mult.dmda";


}

make -C ../../ distclean

cd $BUILDDIR
../../../configure --prefix=$INSTALLDIR --enable-coverage

init;

make clean 1> /dev/null 2> /dev/null
make check
make install -j 1> /dev/null 2> log

apps;

../../../configure --prefix=$INSTALLDIR --enable-coverage --enable-data-rw-lock

make check
make install -j 1> /dev/null 2> log

apps;

generatehtml;
