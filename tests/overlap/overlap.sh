#!/bin/sh -x
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2018       Federal University of Rio Grande do Sul (UFRGS)
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
# Test parsing of FxT traces

# Testing another specific scheduler, no need to run this
[ -z "$STARPU_SCHED" -o "$STARPU_SCHED" = dmdas ] || exit 77

# XXX: Also see examples/mult/sgemm.sh

set -e

PREFIX=$(dirname $0)
rm -rf $PREFIX/overlap.traces
mkdir -p $PREFIX/overlap.traces

export STARPU_FXT_PREFIX=$PREFIX/overlap.traces

$MS_LAUNCHER $STARPU_LAUNCH STARPU_FXT_TRACE=1 STARPU_SCHED=dmdas $PREFIX/overlap
if [ -x $PREFIX/../../tools/starpu_fxt_tool ];
then
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_plot -o $STARPU_FXT_PREFIX -s overlap_sleep_1024_24 -i $STARPU_FXT_PREFIX/prof_file_${USER}_0
	[ -f $STARPU_FXT_PREFIX/starpu_overlap_sleep_1024_24.gp -a -f $STARPU_FXT_PREFIX/starpu_overlap_sleep_1024_24.data -a -f $STARPU_FXT_PREFIX/starpu_overlap_sleep_1024_24_avg.data ]

	# Generate paje, dag, data, etc.
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_tool -d $STARPU_FXT_PREFIX -memory-states -label-deps -i $STARPU_FXT_PREFIX/prof_file_${USER}_0

	$PREFIX/../../tools/starpu_paje_sort $STARPU_FXT_PREFIX/paje.trace
	! type pj_dump || pj_dump -e 0 < $STARPU_FXT_PREFIX/paje.trace

	$PREFIX/../../tools/starpu_codelet_profile $STARPU_FXT_PREFIX/distrib.data overlap_sleep_1024_24
	[ -f $STARPU_FXT_PREFIX/distrib.data.gp -a \( -f $STARPU_FXT_PREFIX/distrib.data.0 -o -f $STARPU_FXT_PREFIX/distrib.data.1 -o -f $STARPU_FXT_PREFIX/distrib.data.2 -o -f $STARPU_FXT_PREFIX/distrib.data.3 -o -f $STARPU_FXT_PREFIX/distrib.data.4 -o -f $STARPU_FXT_PREFIX/distrib.data.5 -o -f $STARPU_FXT_PREFIX/distrib.data.6 \) ]

	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_data_trace -d $STARPU_FXT_PREFIX $STARPU_FXT_PREFIX/prof_file_${USER}_0 overlap_sleep_1024_24
	[ -f $STARPU_FXT_PREFIX/data_trace.gp ]

	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_stats -i $STARPU_FXT_PREFIX/prof_file_${USER}_0
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_tasks_rec_complete $STARPU_FXT_PREFIX/tasks.rec $STARPU_FXT_PREFIX/tasks2.rec
	python3 $PREFIX/../../tools/starpu_trace_state_stats.py $STARPU_FXT_PREFIX/trace.rec
	! type gnuplot || ( $PREFIX/../../tools/starpu_workers_activity -d $STARPU_FXT_PREFIX $STARPU_FXT_PREFIX/activity.data && [ -f $STARPU_FXT_PREFIX/activity.eps ] )

	# needs some R packages
	$PREFIX/../../tools/starpu_paje_draw_histogram $STARPU_FXT_PREFIX/paje.trace || true
	$PREFIX/../../tools/starpu_paje_state_stats $STARPU_FXT_PREFIX/paje.trace || true
	$PREFIX/../../tools/starpu_paje_summary $STARPU_FXT_PREFIX/paje.trace || true
	$PREFIX/../../tools/starpu_codelet_histo_profile $STARPU_FXT_PREFIX/distrib.data || true
	[ -f $STARPU_FXT_PREFIX/distrib.data.overlap_sleep_1024_24.0.a3d3725e.1024.pdf ] || true

	if [ -x $PREFIX/../../tools/starpu_replay ]; then
		$STARPU_LAUNCH $PREFIX/../../tools/starpu_replay $STARPU_FXT_PREFIX/tasks.rec
	fi

	[ ! -x $PREFIX/../../tools/starpu_perfmodel_recdump ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_recdump $STARPU_FXT_PREFIX/tasks.rec -o $STARPU_FXT_PREFIX/perfs2.rec
	[ -f $STARPU_FXT_PREFIX/perfs2.rec ]
fi

[ ! -x $PREFIX/../../tools/starpu_perfmodel_display ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_display -s overlap_sleep_1024_24
[ ! -x $PREFIX/../../tools/starpu_perfmodel_display ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_display -x -s overlap_sleep_1024_24
[ ! -x $PREFIX/../../tools/starpu_perfmodel_recdump ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_recdump -o $STARPU_FXT_PREFIX/perfs.rec
[ -f $STARPU_FXT_PREFIX/perfs.rec ]
