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
	include/starpu-data-filters.h		\
	include/starpu-data-interfaces.h	\
	include/starpu-task.h			\
	include/starpu-data.h			\
	include/starpu-perfmodel.h		\
	include/starpu-util.h			\
	include/starpu-expert.h
