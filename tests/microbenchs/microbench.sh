# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# This provides a helper function to be used for microbenchs that should be run
# under the various schedulers.
#
# The caller should fill either the XFAIL or XSUCCESS variable with the list of
# schedulers which are supposed to fail or succeed, and then call test_scheds

set -e

# disable core generation
ulimit -c 0

# Testing a specific scheduler
if [ -n "$STARPU_SCHED" ]
then
	SCHEDS=$STARPU_SCHED
else
	SCHEDS=`$(dirname $0)/../../tools/starpu_sched_display`
fi

run()
{
	sched=$1

	set +e
	STARPU_SCHED=$sched $STARPU_SUB_PARALLEL $STARPU_LAUNCH $(dirname $0)/$TEST "$@"
	ret=$?
	set -e
	if test $ret = 0
	then
		( echo PASS: STARPU_SCHED=$sched ./microbenchs/$TEST >&9 ) 2> /dev/null || true
		echo "SUCCESS: STARPU_SCHED=$sched ./microbenchs/$TEST"
		return 0
	fi
	if test $ret = 77
	then
		echo "SKIP: STARPU_SCHED=$sched ./microbenchs/$TEST"
		return 0
	fi

	RESULT=0
	if [ -n "$XSUCCESS" ]
	then
		# We have a list of schedulers that are expected to
		# succeed, others are allowed to fail
		case " $XSUCCESS " in
			*\ $sched\ *)
				echo "FAIL: STARPU_SCHED=$sched ./microbenchs/$TEST" | ( tee /dev/tty || true )
				RESULT=1
				;;
			*)
				echo "XFAIL: STARPU_SCHED=$sched ./microbenchs/$TEST"
				;;
		esac
	else
		# We have a list of schedulers that are expected to
		# fail, others are expected to succeed
		case " $XFAIL " in
			*\ $sched\ *)
				echo "XFAIL: STARPU_SCHED=$sched ./microbenchs/$TEST"
				;;
			*)
				echo "FAIL: STARPU_SCHED=$sched ./microbenchs/$TEST" | ( tee /dev/tty || true )
				RESULT=1
				;;
		esac
	fi
	return $RESULT
}

test_scheds()
{
	TEST=$1
	shift

	RESULT=0
	if [ -n "$STARPU_SUB_PARALLEL" ]
	then
		for sched in $SCHEDS
		do
			run $sched &
		done
		while true
		do
			set +e
			wait -n
			RET=$?
			set -e
			if [ $RET = 127 ] ; then break ; fi
			if [ $RET != 0 -a $RET != 77 ] ; then RESULT=1 ; fi
		done
	else
		for sched in $SCHEDS
		do
			set +e
			run $sched
			RET=$?
			set -e
			if [ $RET != 0 -a $RET != 77 ] ; then RESULT=1 ; fi
		done
	fi
	exit $RESULT
}
