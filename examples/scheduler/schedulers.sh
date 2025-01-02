#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2010-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	( echo FAIL: STARPU_SCHED=$sched $basedir/../cholesky/cholesky_tag >&9 ) 2> /dev/null || true
	echo "failure" >&2
        exit $1
    else
	( echo PASS: STARPU_SCHED=$sched $basedir/../cholesky/cholesky_tag >&9 ) 2> /dev/null || true
    fi
}

basedir=$(dirname $0)
if test ! -x $basedir/../cholesky/cholesky_tag
then
    echo "Application $basedir/../cholesky/cholesky_tag unavailable"
    exit 77
fi

if [ -n "$STARPU_SCHED" ]
then
	SCHEDULERS=$STARPU_SCHED
else
	SCHEDULERS=`$basedir/../../tools/starpu_sched_display | grep -v heteroprio`
fi

if [ "$STARPU_QUICK_CHECK" = 1 ]
then
	SIDE=32
else
	SIDE=320
fi

run()
{
    sched=$1
    echo "cholesky.$sched"
    STARPU_SCHED=$sched $STARPU_SUB_PARALLEL $MS_LAUNCHER $STARPU_LAUNCH $basedir/../cholesky/cholesky_tag -size $(($SIDE*3)) -nblocks 3
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
