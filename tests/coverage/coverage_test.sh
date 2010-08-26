#!/bin/bash
#
# StarPU
# Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
WORKDIR=`mktemp -d`
COVDIR=$WORKDIR/coverage
BUILDDIR=$WORKDIR/build
INSTALLDIR=$WORKDIR/local
EXAMPLEDIR=$INSTALLDIR/lib/starpu/examples/
LOGFILE=`mktemp`

init()
{
    mkdir -p $INSTALLDIR
    mkdir -p $BUILDDIR
    mkdir -p $COVDIR
    lcov --directory $BUILDDIR --zerocounters >$LOGFILE 2>&1
}

save_cov()
{
    testname=$1
    lcov --directory $BUILDDIR --capture --output $COVDIR/$testname.info >>$LOGFILE 2>&1
    lcov -a $COVDIR/$testname.info -o $COVDIR/all.info >>$LOGFILE 2>&1
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
    $EXAMPLEDIR/incrementer >>$LOGFILE 2>&1
    save_cov "incrementer";

    echo "tag_example"
    $EXAMPLEDIR/tag_example -iter 64 -i 128 -j 24 >>$LOGFILE 2>&1
    save_cov "tag_example";

    echo "tag_example2"
    $EXAMPLEDIR/tag_example2 -iter 64 -i 128 >>$LOGFILE 2>&1
    save_cov "tag_example2";

    #echo "spmv"
    #$BUILDDIR/examples/spmv/dw_spmv >>$LOGFILE 2>&1
    #save_cov "spmv";
    #
    #echo "spmv.gpu"
    #STARPU_NCPUS=0 $BUILDDIR/examples/spmv/dw_spmv >>$LOGFILE 2>&1
    #save_cov "spmv.gpu";
    #
    #echo "spmv.cpu"
    #STARPU_NCUDA=0 $BUILDDIR/examples/spmv/dw_spmv >>$LOGFILE 2>&1
    #save_cov "spmv.cpu";
    #
    #echo "spmv.dm"
    #STARPU_SCHED="dm" $BUILDDIR/examples/spmv/dw_spmv >>$LOGFILE 2>&1
    #save_cov "spmv.dm";
    #
    #echo "spmv.dmda"
    #STARPU_SCHED="dmda" $BUILDDIR/examples/spmv/dw_spmv >>$LOGFILE 2>&1
    #save_cov "spmv.dmda";

    #echo "strassen.ws"
    #STARPU_SCHED="ws" $EXAMPLEDIR/dw_strassen -rec 3 -size 2048 -pin >>$LOGFILE 2>&1
    #save_cov "strassen.ws";
    #
    #echo "strassen.dm"
    #STARPU_SCHED="dm" $EXAMPLEDIR/dw_strassen -rec 3 -size 2048 -pin >>$LOGFILE 2>&1
    #save_cov "strassen.dm";
    #
    #echo "strassen.dmda"
    #STARPU_SCHED="dmda" $EXAMPLEDIR/dw_strassen -rec 3 -size 2048 -pin >>$LOGFILE 2>&1
    #save_cov "strassen.dmda";
    
    echo "chol.dm"
    STARPU_CALIBRATE=1 STARPU_SCHED="dm" $EXAMPLEDIR/dw_cholesky -pin >>$LOGFILE 2>&1
    save_cov "chol.dm";

    echo "chol.dmda"
    STARPU_CALIBRATE=1 STARPU_SCHED="dmda" $EXAMPLEDIR/dw_cholesky -pin >>$LOGFILE 2>&1
    save_cov "chol.dmda";

    echo "chol.cpu"
    STARPU_CALIBRATE=1 STARPU_NCUDA=0 STARPU_SCHED="dm" $EXAMPLEDIR/dw_cholesky -pin >>$LOGFILE 2>&1
    save_cov "chol.cpu";

    echo "chol.gpu"
    STARPU_CALIBRATE=1 STARPU_NCPUS=0 STARPU_SCHED="dm" $EXAMPLEDIR/dw_cholesky -pin >>$LOGFILE 2>&1
    save_cov "chol.gpu";

    echo "chol"
    $EXAMPLE/dw_cholesky >>$LOGFILE 2>&1
    save_cov "chol";

    echo "heat.dm.4k.calibrate.v2"
    STARPU_CALIBRATE=1 STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 66 -nblocks 4 -v2 -pin >>$LOGFILE 2>&1
    save_cov "heat.dm.4k.calibrate.v2";

    echo "heat.dm.8k.calibrate.v2"
    STARPU_CALIBRATE=1 STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin >>$LOGFILE 2>&1
    save_cov "heat.dm.8k.calibrate.v2";

    echo "heat.dm.16k.calibrate.v2"
    STARPU_CALIBRATE=1 STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 130 -nthick 130 -nblocks 16 -v2 -pin >>$LOGFILE 2>&1
    save_cov "heat.dm.16k.calibrate.v2";

    echo "heat.dm.8k.no.pin.v2"
    STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 >>$LOGFILE 2>&1
    save_cov "heat.dm.8k.no.pin.v2";

    #echo "heat.prio.8k"
    #STARPU_SCHED="prio" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin >>$LOGFILE 2>&1
    #save_cov "heat.prio.8k";
    
    echo "heat.dm.8k.v2.no.prio"
    STARPU_SCHED="no-prio" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 >>$LOGFILE 2>&1
    save_cov "heat.dm.8k.v2.no.prio";

    echo "heat.dm.8k.v2.random"
    STARPU_SCHED="random" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 >>$LOGFILE 2>&1
    save_cov "heat.dm.8k.v2.random";

    echo "heat.dm.8k.v2"
    STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 >>$LOGFILE 2>&1
    save_cov "heat.dm.8k.v2";

    echo "heat.dm.16k.v2"
    STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 130 -nthick 130 -nblocks 16 -pin -v2 >>$LOGFILE 2>&1
    save_cov "heat.dm.16k.v2";

    #echo "heat.ws.8k.v2"
    #STARPU_SCHED="ws" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 >>$LOGFILE 2>&1
    #save_cov "heat.ws.8k.v2";
    
    echo "heat.greedy.8k.v2"
    STARPU_SCHED="greedy" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 >>$LOGFILE 2>&1
    save_cov "heat.greedy.8k.v2";

    echo "heat.8k.cg"
    $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg >>$LOGFILE 2>&1
    save_cov "heat.8k.cg";

    echo "heat.dm.8k.cg"
    STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg >>$LOGFILE 2>&1
    save_cov "heat.dm.8k.cg";

    #echo "heat.dm.8k.v3"
    #STARPU_SCHED="dm" $EXAMPLEDIR/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v3 >>$LOGFILE 2>&1
    #save_cov "heat.dm.8k.v3";
    
    echo "mult.dm.common"
    STARPU_SCHED="dm" $EXAMPLEDIR/dw_mult -nblocks 4 -x 4096 -y 4096 -z 1024 -pin -common-model >>$LOGFILE 2>&1
    save_cov "mult.dm.common";

    echo "mult.dm"
    STARPU_CALIBRATE=1 STARPU_SCHED="dm" $EXAMPLEDIR/dw_mult -nblocks 8 -x 8192 -y 8192 -z 8192 -pin >>$LOGFILE 2>&1
    save_cov "mult.dm";

    echo "mult.dmda"
    STARPU_CALIBRATE=1 STARPU_SCHED="dmda" $EXAMPLEDIR/dw_mult -nblocks 8 -x 8192 -y 8192 -z 8192 -pin >>$LOGFILE 2>&1
    save_cov "mult.dmda";
}

init;

cd $BUILDDIR
$ROOTDIR/configure --enable-coverage --prefix=$INSTALLDIR >>$LOGFILE 2>&1
make clean >>$LOGFILE 2>&1
make check
make install -j >>$LOGFILE 2>&1
apps;

$ROOTDIR/configure --prefix=$INSTALLDIR --enable-coverage --enable-data-rw-lock
make clean 1> /dev/null 2> /dev/null
make check
make install -j 1> /dev/null 2> log
apps;

generatehtml;

echo
echo "See $WORKDIR and $LOGFILE for detailed output"
echo
