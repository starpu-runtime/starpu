#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

if test -n "$STARPU_MICROBENCHS_DISABLED" ; then exit 77 ; fi

ROOT=${0%.sh}
ROOT=${ROOT%_sched}
unset STARPU_SSILENT
$_MS_LAUNCHER $STARPU_LAUNCH $_STARPU_LAUNCH $ROOT "$@" > tasks_size_overhead.output
ret=$?
if test "$ret" = "0" && [ -z "$(echo $MAKEFLAGS | sed -ne 's/.*-j\([0-9]\+\).*/\1/p')" ]
then
    # if the program was successful and we are not running in parallel, try to run gnuplot
    DIR=
    [ -z "$STARPU_BENCH_DIR" ] || DIR="$STARPU_BENCH_DIR/"
    export TERMINAL=png
    export OUTFILE=${DIR}tasks_size_overhead_${STARPU_SCHED}.png
    gnuplot_av=$(command -v gnuplot)
    if test -n "$gnuplot_av"
    then
	# If gnuplot is available, plot the result
	$ROOT.gp
	ret=$?
    fi
fi

exit $ret
