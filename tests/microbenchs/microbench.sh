# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# Testing a specific scheduler
if [ -n "$STARPU_SCHED" ]
then
	SCHEDS=$STARPU_SCHED
else
	SCHEDS=`$(dirname $0)/../../tools/starpu_sched_display`
fi

test_scheds()
{
	TEST=$1
	shift
	xfailed=""
	failed=""
	pass=""
	skip=""

	if [ -n "$STARPU_MIC_SINK_PROGRAM_PATH" ] ; then
		STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/$TEST
		# in case libtool got into play
		[ -x "$STARPU_MIC_SINK_PROGRAM_PATH/.libs/$TEST" ] && STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/.libs/$TEST
	fi

	RESULT=0
	for sched in $SCHEDS;
	do
	    	set +e
		STARPU_SCHED=$sched $STARPU_LAUNCH $(dirname $0)/$TEST "$@"
		ret=$?
	    	set -e
		if test $ret = 0
		then
		    	echo "SUCCESS: STARPU_SCHED=$sched ./microbenchs/$TEST"
			pass="$pass $sched"
			continue
		fi
		if test $ret = 77
		then
		    	echo "SKIP: STARPU_SCHED=$sched ./microbenchs/$TEST"
			skip="$skip $sched"
			continue
		fi

		if [ -n "$XSUCCESS" ]
		then
                        # We have a list of schedulers that are expected to
                        # succeed, others are allowed to fail
			case " $XSUCCESS " in
				*\ $sched\ *)
					echo "FAIL: STARPU_SCHED=$sched ./microbenchs/$TEST" | ( tee /dev/tty || true )
					failed="$failed $sched"
					RESULT=1
					;;
				*)
					echo "XFAIL: STARPU_SCHED=$sched ./microbenchs/$TEST"
					xfailed="$xfailed $sched"
					;;
			esac
		else
                        # We have a list of schedulers that are expected to
                        # fail, others are expected to succeed
			case " $XFAIL " in
				*\ $sched\ *)
					echo "XFAIL: STARPU_SCHED=$sched ./microbenchs/$TEST"
					xfailed="$xfailed $sched"
					;;
				*)
					echo "FAIL: STARPU_SCHED=$sched ./microbenchs/$TEST" | ( tee /dev/tty || true )
					failed="$failed $sched"
					RESULT=1
					;;
			esac
		fi

	done
	echo "passed schedulers:$pass"| ( tee /dev/tty || true )
	echo "skipped schedulers:$skip"| ( tee /dev/tty || true )
	echo "failed schedulers:$failed"| ( tee /dev/tty || true )
	echo "xfailed schedulers:$xfailed"| ( tee /dev/tty || true )
	return $RESULT
}
