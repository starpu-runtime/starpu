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

# XXX: Also see tests/overlap/overlap.sh

set -e

PREFIX=$(dirname $0)
rm -rf $PREFIX/sgemm.traces
mkdir -p $PREFIX/sgemm.traces

export STARPU_FXT_PREFIX=$PREFIX/sgemm.traces

STARPU_FXT_TRACE=1 STARPU_SCHED=dmdas $MS_LAUNCHER $STARPU_LAUNCH $PREFIX/sgemm -check
if [ -x $PREFIX/../../tools/starpu_fxt_tool ];
then
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_plot -o $STARPU_FXT_PREFIX -s starpu_sgemm_gemm -i $STARPU_FXT_PREFIX/prof_file_${USER}_0
	[ -f $STARPU_FXT_PREFIX/starpu_starpu_sgemm_gemm.gp -a -f $STARPU_FXT_PREFIX/starpu_starpu_sgemm_gemm.data -a -f $STARPU_FXT_PREFIX/starpu_starpu_sgemm_gemm.data ]

	# Generate paje, dag, data, etc.
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_tool -d $STARPU_FXT_PREFIX -memory-states -label-deps -i $STARPU_FXT_PREFIX/prof_file_${USER}_0

	$PREFIX/../../tools/starpu_paje_sort $STARPU_FXT_PREFIX/paje.trace
	! type pj_dump || pj_dump -e 0 < $STARPU_FXT_PREFIX/paje.trace

	$PREFIX/../../tools/starpu_codelet_profile $STARPU_FXT_PREFIX/distrib.data starpu_sgemm_gemm
	[ -f $STARPU_FXT_PREFIX/distrib.data.gp -a \( -f $STARPU_FXT_PREFIX/distrib.data.0 -o -f $STARPU_FXT_PREFIX/distrib.data.1 -o -f $STARPU_FXT_PREFIX/distrib.data.2 -o -f $STARPU_FXT_PREFIX/distrib.data.3 -o -f $STARPU_FXT_PREFIX/distrib.data.4 -o -f $STARPU_FXT_PREFIX/distrib.data.5 -o -f $STARPU_FXT_PREFIX/distrib.data.6 \) ]

	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_data_trace -d $STARPU_FXT_PREFIX $STARPU_FXT_PREFIX/prof_file_${USER}_0 starpu_sgemm_gemm
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
	[ -f $STARPU_FXT_PREFIX/distrib.data.starpu_sgemm_gemm.0.492beed5.33177600.pdf ] || true

	if [ -x $PREFIX/../../tools/starpu_replay ]; then
		$STARPU_LAUNCH $PREFIX/../../tools/starpu_replay $STARPU_FXT_PREFIX/tasks.rec
	fi

	[ ! -x $PREFIX/../../tools/starpu_perfmodel_recdump ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_recdump $STARPU_FXT_PREFIX/tasks.rec -o $STARPU_FXT_PREFIX/perfs2.rec
	[ -f $STARPU_FXT_PREFIX/perfs2.rec ]
fi

[ ! -x $PREFIX/../../tools/starpu_perfmodel_display ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_display -s starpu_sgemm_gemm
[ ! -x $PREFIX/../../tools/starpu_perfmodel_display ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_display -x -s starpu_sgemm_gemm
[ ! -x $PREFIX/../../tools/starpu_perfmodel_recdump ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_recdump -o $STARPU_FXT_PREFIX/perfs.rec
[ -f $STARPU_FXT_PREFIX/perfs.rec ]
