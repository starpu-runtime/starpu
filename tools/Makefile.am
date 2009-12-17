#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

LIBS = $(top_builddir)/src/libstarpu.la @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/

bin_PROGRAMS =

CLEANFILES = *.gcno *.gcda *.linkinfo

if USE_FXT
bin_PROGRAMS += fxt-tool fxt-stats

fxt_tool_SOURCES = fxt-tool.c dag-dot.c histo-paje.c
fxt_tool_CFLAGS = -I$(top_srcdir)/src/
fxt_tool_LDADD = 

fxt_stats_SOURCES = fxt-stats.c
fxt_stats_CFLAGS = -I$(top_srcdir)/src/
endif

bin_PROGRAMS +=	perfmodel-display
perfmodel_display_SOURCES = perfmodel-display.c
