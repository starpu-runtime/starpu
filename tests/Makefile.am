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

LIBS = $(top_builddir)/src/libstarpu.la @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/

SUBDIRS = 

TESTS = $(check_PROGRAMS)

check_PROGRAMS =

check_PROGRAMS += 				\
	microbenchs/async-tasks-overhead	\
	microbenchs/sync-tasks-overhead		\
	microbenchs/tag-wait-api

microbenchs_async_tasks_overhead_SOURCES =	\
	microbenchs/async-tasks-overhead.c

microbenchs_sync_tasks_overhead_SOURCES =	\
	microbenchs/sync-tasks-overhead.c

microbenchs_tag_wait_api_SOURCES =		\
	microbenchs/tag-wait-api.c

