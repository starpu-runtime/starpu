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

ACLOCAL_AMFLAGS=-I m4
CLEANFILES = *.gcno *.gcda *.linkinfo

SUBDIRS = src
if USE_MPI
SUBDIRS += mpi
endif
SUBDIRS += tools examples tests doc

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
	include/starpu_bound.h			\
	include/starpu_scheduler.h

txtdir = ${prefix}
txt_DATA = AUTHORS COPYING.LGPL README
EXTRA_DIST = AUTHORS COPYING.LGPL README


