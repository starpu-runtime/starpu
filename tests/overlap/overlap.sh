#!/bin/sh -x
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017                                     CNRS
# Copyright (C) 2017,2018-2020                           Université de Bordeaux
# Copyright (C) 2017                                     Inria
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

set -e

PREFIX=$(dirname $0)
STARPU_SCHED=dmdas STARPU_FXT_PREFIX=$PREFIX/ $PREFIX/overlap
[ ! -x $PREFIX/../../tools/starpu_perfmodel_display ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_display -s overlap_sleep_1024_24
[ ! -x $PREFIX/../../tools/starpu_perfmodel_plot -o ! -f $PREFIX/prof_file_${USER}_0 ] || $STARPU_LAUNCH $PREFIX/../../tools/starpu_perfmodel_plot -s overlap_sleep_1024_24 -i $PREFIX/prof_file_${USER}_0
if [ -x $PREFIX/../../tools/starpu_fxt_tool ];
then
	# Generate paje, dag, data, etc.
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_tool -i $PREFIX/prof_file_${USER}_0

	$PREFIX/../../tools/starpu_paje_sort paje.trace

	$PREFIX/../../tools/starpu_codelet_profile distrib.data overlap_sleep_1024_24
	[ -f distrib.data.gp -a \( -f distrib.data.0 -o -f distrib.data.1 -o -f distrib.data.2 -o -f distrib.data.3 -o -f distrib.data.4 \) ]

	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_data_trace $PREFIX/prof_file_${USER}_0 overlap_sleep_1024_24
	[ -f data_trace.gp ]

	$STARPU_LAUNCH $PREFIX/../../tools/starpu_fxt_stats -i $PREFIX/prof_file_${USER}_0
	$STARPU_LAUNCH $PREFIX/../../tools/starpu_tasks_rec_complete tasks.rec tasks2.rec
	$PREFIX/../../tools/starpu_workers_activity activity.data
	[ -f activity.eps ]

	# needs some R packages
	$PREFIX/../../tools/starpu_paje_draw_histogram paje.trace || true
	$PREFIX/../../tools/starpu_paje_state_stats paje.trace || true
	$PREFIX/../../tools/starpu_paje_summary paje.trace || true
	$PREFIX/../../tools/starpu_codelet_histo_profile distrib.data || true
	[ -f distrib.data.overlap_sleep_1024_24.0.a3d3725e.1024.pdf ] || true
fi
