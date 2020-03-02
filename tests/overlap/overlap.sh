#!/bin/sh -x
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

if [ -n "$STARPU_MIC_SINK_PROGRAM_PATH" ] ; then
	STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/overlap
	# in case libtool got into play
	[ -x "$STARPU_MIC_SINK_PROGRAM_PATH/.libs/overlap" ] && STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/.libs/overlap
fi

STARPU_SCHED=dmdas STARPU_FXT_PREFIX=$PREFIX/ $PREFIX/overlap
[ ! -x $PREFIX/../../tools/starpu_perfmodel_display ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_display -s overlap_sleep_1024_24
[ ! -x $PREFIX/../../tools/starpu_perfmodel_display ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_display -x -s overlap_sleep_1024_24
[ ! -x $PREFIX/../../tools/starpu_perfmodel_recdump ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_recdump -o perfs.rec
[ -f perfs.rec ]
if [ -x $PREFIX/../../tools/starpu_fxt_tool ];
then
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_plot -s overlap_sleep_1024_24 -i $PREFIX/prof_file_${USER}_0
	[ -f starpu_overlap_sleep_1024_24.gp -a -f starpu_overlap_sleep_1024_24.data -a -f starpu_overlap_sleep_1024_24_avg.data ]

	# Generate paje, dag, data, etc.
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_tool -memory-states -label-deps -i $PREFIX/prof_file_${USER}_0

	$PREFIX/../../tools/starpu_paje_sort paje.trace
	! type pj_dump || pj_dump -e 0 < paje.trace

	$PREFIX/../../tools/starpu_codelet_profile distrib.data overlap_sleep_1024_24
	[ -f distrib.data.gp -a \( -f distrib.data.0 -o -f distrib.data.1 -o -f distrib.data.2 -o -f distrib.data.3 -o -f distrib.data.4 \) ]

	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_data_trace $PREFIX/prof_file_${USER}_0 overlap_sleep_1024_24
	[ -f data_trace.gp ]

	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_stats -i $PREFIX/prof_file_${USER}_0
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_tasks_rec_complete tasks.rec tasks2.rec
	python $PREFIX/../../tools/starpu_trace_state_stats.py trace.rec
	$PREFIX/../../tools/starpu_workers_activity activity.data
	[ -f activity.eps ]

	# needs some R packages
	$PREFIX/../../tools/starpu_paje_draw_histogram paje.trace || true
	$PREFIX/../../tools/starpu_paje_state_stats paje.trace || true
	$PREFIX/../../tools/starpu_paje_summary paje.trace || true
	$PREFIX/../../tools/starpu_codelet_histo_profile distrib.data || true
	[ -f distrib.data.overlap_sleep_1024_24.0.a3d3725e.1024.pdf ] || true

	if [ -x $PREFIX/../../tools/starpu_replay ]; then
		$STARPU_LAUNCH $PREFIX/../../tools/starpu_replay tasks.rec
	fi

	[ ! -x $PREFIX/../../tools/starpu_perfmodel_recdump ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_recdump tasks.rec -o perfs2.rec
	[ -f perfs2.rec ]
fi

