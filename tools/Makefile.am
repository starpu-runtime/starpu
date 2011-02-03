# StarPU --- Runtime system for heterogeneous multicore architectures.
# 
# Copyright (C) 2009, 2010  Université de Bordeaux 1
# Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

SUBDIRS = 

AM_CFLAGS = $(HWLOC_CFLAGS)
LIBS = $(top_builddir)/src/libstarpu.la $(HWLOC_LIBS) @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_builddir)/src -I$(top_srcdir)/src

bin_PROGRAMS =
dist_bin_SCRIPTS =

CLEANFILES = *.gcno *.gcda *.linkinfo

bin_PROGRAMS += starpu_calibrate_bus

starpu_calibrate_bus_SOURCES = starpu_calibrate_bus.c

if STARPU_USE_FXT
# TODO: prefix with starpu_
bin_PROGRAMS += starpu_fxt_tool starpu_fxt_stats

starpu_fxt_tool_SOURCES = starpu_fxt_tool.c fxt_tool_common.c fxt_tool_mpi.c dag_dot.c histo_paje.c
starpu_fxt_tool_CFLAGS = -I$(top_srcdir)/src/
starpu_fxt_tool_LDADD = 

starpu_fxt_stats_SOURCES = starpu_fxt_stats.c
starpu_fxt_stats_CFLAGS = -I$(top_srcdir)/src/
endif

bin_PROGRAMS +=	starpu_perfmodel_display
starpu_perfmodel_display_SOURCES = starpu_perfmodel_display.c

noinst_PROGRAMS =	cbc2paje lp2paje

noinst_HEADERS = \
	fxt_tool.h	\
	histo_paje.h

dist_bin_SCRIPTS +=	\
	starpu_top.sh
