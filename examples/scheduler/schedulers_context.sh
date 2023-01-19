#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
check_success()
{
    if [ $1 -ne 0 ] ; then
	( echo FAIL: STARPU_SCHED=$sched $basedir/../sched_ctx/sched_ctx >&9 ) 2> /dev/null || true
	echo "failure" >&2
        exit $1
    else
	( echo PASS: STARPU_SCHED=$sched $basedir/../sched_ctx/sched_ctx >&9 ) 2> /dev/null || true
    fi
}

basedir=$(dirname $0)
if test ! -x $basedir/../sched_ctx/sched_ctx
then
    echo "Application $basedir/../sched_ctx/sched_ctx unavailable"
    exit 77
fi

if [ -n "$STARPU_SCHED" ]
then
	SCHEDULERS="$STARPU_SCHED"
else
	SCHEDULERS=`$basedir/../../tools/starpu_sched_display | grep -v pheft | grep -v peager | grep -v heteroprio | grep -v modular-gemm`
fi

run()
{
    sched=$1
    echo "sched_ctx.$sched"
    STARPU_SCHED=$sched $STARPU_SUB_PARALLEL $STARPU_LAUNCH $basedir/../sched_ctx/sched_ctx
    check_success $?
}

if [ -n "$STARPU_SUB_PARALLEL" ]
then
	for sched in $SCHEDULERS
	do
		run $sched &
	done
	RESULT=0
	while true
	do
		wait -n
		RET=$?
		if [ $RET = 127 ] ; then break ; fi
		if [ $RET != 0 -a $RET != 77 ] ; then RESULT=1 ; fi
	done
	exit $RESULT
else
	for sched in $SCHEDULERS
	do
		run $sched
	done
fi
