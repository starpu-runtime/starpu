#
# StarPU
# Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

SUBDIRS = 

AM_CFLAGS = $(HWLOC_CFLAGS)
LIBS = $(top_builddir)/src/libstarpu.la $(HWLOC_LIBS) @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_builddir)/src -I$(top_srcdir)/src

bin_PROGRAMS =
dist_bin_SCRIPTS =

CLEANFILES = *.gcno *.gcda *.linkinfo

bin_PROGRAMS += calibrate_bus

calibrate_bus_SOURCES = calibrate_bus.c

if STARPU_USE_FXT
bin_PROGRAMS += fxt_tool fxt_stats

fxt_tool_SOURCES = fxt_tool.c fxt_tool_common.c fxt_tool_mpi.c dag_dot.c histo_paje.c
fxt_tool_CFLAGS = -I$(top_srcdir)/src/
fxt_tool_LDADD = 

fxt_stats_SOURCES = fxt_stats.c
fxt_stats_CFLAGS = -I$(top_srcdir)/src/
endif

bin_PROGRAMS +=	perfmodel_display
perfmodel_display_SOURCES = perfmodel_display.c

noinst_PROGRAMS +=	cbc2paje lp2paje

noinst_HEADERS = \
	fxt_tool.h	\
	histo_paje.h

dist_bin_SCRIPTS +=	\
	starpu_top.sh
