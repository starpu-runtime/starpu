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

SCHEDS=`$top_builddir/tools/starpu_sched_display`

test_scheds()
{
	TEST=$1

	RESULT=0
	for sched in $SCHEDS;
	do
<<<<<<< HEAD
		if STARPU_SCHED=$sched $top_builddir/tests/microbenchs/$TEST ; then
			echo " SUCCESS: STARPU_SCHED=$sched ./microbenchs/$TEST"
=======
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
>>>>>>> f2bea4ce9... Use $STARPU_LAUNCH in scripts
			continue
		fi

		if [ -n "$XSUCCESS" ]
		then
                        # We have a list of schedulers that are expected to
                        # succeed, others are allowed to fail
			case " $XSUCCESS " in 
				*\ $sched\ *)
					echo " FAIL: STARPU_SCHED=$sched ./microbenchs/$TEST" | ( tee /dev/tty || true )
					RESULT=1
					;;
				*)
					echo " XFAIL: STARPU_SCHED=$sched ./microbenchs/$TEST"
					;;
			esac
		else
                        # We have a list of schedulers that are expected to
                        # fail, others are expected to succeed
			case " $XFAIL " in 
				*\ $sched\ *)
					echo " XFAIL: STARPU_SCHED=$sched ./microbenchs/$TEST"
					;;
				*)
					echo " FAIL: STARPU_SCHED=$sched ./microbenchs/$TEST" | ( tee /dev/tty || true )
					RESULT=1
					;;
			esac
		fi

	done
	return $RESULT
}
