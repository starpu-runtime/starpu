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

check_success()
{
    if [ $1 != 0 ] ; then
        exit $1
    fi
}

apps()
{
    echo "block opencl"
    STARPU_NCUDA=0 STARPU_NCPUS=0 ./basic_examples/block
    check_success $?

    echo "variable opencl"
    STARPU_NCUDA=0 STARPU_NCPUS=0 ./basic_examples/variable 100
    check_success $?

    echo "variable no worker"
    STARPU_NCUDA=0 STARPU_NOPENCL=0 STARPU_NCPUS=0 ./basic_examples/variable 
    check_success $?

    echo "incrementer opencl"
    STARPU_NCUDA=0 STARPU_NCPUS=0 ./incrementer/incrementer 10
    check_success $?

    echo "incrementer no worker"
    STARPU_NCUDA=0 STARPU_NOPENCL=0 STARPU_NCPUS=0 ./incrementer/incrementer
    check_success $?

    echo "tag_example"
    ./tag_example/tag_example -iter 64 -i 128 -j 24
    check_success $?

    echo "tag_example2"
    ./tag_example/tag_example2 -iter 64 -i 128
    check_success $?

    if [ -f ./cholesky/dw_cholesky ] ; then
	echo "chol.dm"
	STARPU_CALIBRATE=1 STARPU_SCHED="dm" ./cholesky/dw_cholesky -pin
	check_success $?

	echo "chol.dmda"
	STARPU_CALIBRATE=1 STARPU_SCHED="dmda" ./cholesky/dw_cholesky -pin
	check_success $?

	echo "chol.cpu"
	STARPU_CALIBRATE=1 STARPU_NCUDA=0 STARPU_SCHED="dm" ./cholesky/dw_cholesky -pin
	check_success $?

	echo "chol.gpu"
	STARPU_CALIBRATE=1 STARPU_NCPUS=0 STARPU_SCHED="dm" ./cholesky/dw_cholesky -pin
	check_success $?
    fi

    if [ -f ./heat/heat ] ; then
	echo "heat.dm.4k.calibrate.v2"
	STARPU_CALIBRATE=1 STARPU_SCHED="dm" ./heat/heat -ntheta 66 -nthick 66 -nblocks 4 -v2 -pin
	check_success $?

	echo "heat.dm.8k.calibrate.v2"
	STARPU_CALIBRATE=1 STARPU_SCHED="dm" ./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -v2 -pin
	check_success $?

	echo "heat.dm.8k.no.pin.v2"
	STARPU_SCHED="dm" ./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -v2
	check_success $?

	echo "heat.dm.8k.v2.no.prio"
	STARPU_SCHED="no-prio" ./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2
	check_success $?

	echo "heat.dm.8k.v2.random"
	STARPU_SCHED="random" ./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2
	check_success $?

	echo "heat.dm.8k.v2"
	STARPU_SCHED="dm" ./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2
	check_success $?

	echo "heat.greedy.8k.v2"
	STARPU_SCHED="greedy" ./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2
	check_success $?

	echo "heat.8k.cg"
	./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg
	check_success $?

	echo "heat.dm.8k.cg"
	STARPU_SCHED="dm" ./heat/heat -ntheta 66 -nthick 130 -nblocks 8 -pin -v2 -cg
	check_success $?
    fi

    if [ -f ./mult/dw_mult_no_stride ] ; then
	echo "mult.dm.common"
	STARPU_SCHED="dm" ./mult/dw_mult_no_stride -nblocks 4 -x 4096 -y 4096 -z 1024 -pin -common-model
	check_success $?

	echo "mult.dm"
	STARPU_CALIBRATE=1 STARPU_SCHED="dm" ./mult/dw_mult_no_stride -nblocks 8 -x 8192 -y 8192 -z 8192 -pin
	check_success $?

	echo "mult.dmda"
	STARPU_CALIBRATE=1 STARPU_SCHED="dmda" ./mult/dw_mult_no_stride -nblocks 8 -x 8192 -y 8192 -z 8192 -pin
	check_success $?
    fi
}

apps;
