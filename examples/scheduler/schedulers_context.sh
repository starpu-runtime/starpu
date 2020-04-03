#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	echo "failure" >&2
        exit $1
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

for sched in $SCHEDULERS
do
    echo "sched_ctx.$sched"
    STARPU_SCHED=$sched $STARPU_LAUNCH $basedir/../sched_ctx/sched_ctx
    check_success $?
done
