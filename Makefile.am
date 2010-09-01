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

ACLOCAL_AMFLAGS=-I m4
CLEANFILES = *.gcno *.gcda *.linkinfo

SUBDIRS = src tools examples tests doc

if USE_MPI
SUBDIRS += mpi
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libstarpu.pc

include_HEADERS = 				\
	include/starpu.h			\
	include/starpu_config.h			\
	include/starpu_data_filters.h		\
	include/starpu_data_interfaces.h	\
	include/starpu_task.h			\
	include/starpu_task_list.h		\
	include/starpu_data.h			\
	include/starpu_perfmodel.h		\
	include/starpu_util.h			\
	include/starpu_cuda.h			\
	include/starpu_opencl.h			\
	include/starpu_expert.h			\
	include/starpu_profiling.h		\
	include/starpu_scheduler.h

noinst_HEADERS = \
	include/pthread_win32/pthread.h		\
	include/pthread_win32/semaphore.h
