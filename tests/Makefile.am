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

CLEANFILES = *.gcno *.gcda *.linkinfo

if USE_CUDA

# TODO define NVCCFLAGS
NVCC ?= nvcc

NVCCFLAGS += -I$(top_srcdir)/include/

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS)

.cu.o:
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir}

endif


SUBDIRS = 

TESTS = $(check_PROGRAMS)

check_PROGRAMS =

check_PROGRAMS += 				\
	microbenchs/async-tasks-overhead	\
	microbenchs/sync-tasks-overhead		\
	microbenchs/tag-wait-api		\
	microbenchs/sync_and_notify_data

microbenchs_async_tasks_overhead_SOURCES =	\
	microbenchs/async-tasks-overhead.c

microbenchs_sync_tasks_overhead_SOURCES =	\
	microbenchs/sync-tasks-overhead.c

microbenchs_tag_wait_api_SOURCES =		\
	microbenchs/tag-wait-api.c

if USE_CUDA

microbenchs_sync_and_notify_data_SOURCES =	\
	microbenchs/sync_and_notify_data.c	\
	microbenchs/sync_and_notify_data_kernels.cu

else

microbenchs_sync_and_notify_data_SOURCES =	\
	microbenchs/sync_and_notify_data.c

endif
