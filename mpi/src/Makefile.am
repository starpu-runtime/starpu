# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2012  Université de Bordeaux 1
# Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

CC=$(MPICC)
CCLD=$(MPICC)

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/src/ -I$(top_builddir)/src -I$(top_builddir)/include -I$(top_srcdir)/mpi/include -I$(top_srcdir)/mpi/src
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS)

lib_LTLIBRARIES = libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

libstarpumpi_@STARPU_EFFECTIVE_VERSION@_la_LIBADD = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la
libstarpumpi_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS = $(ldflags) -no-undefined					\
  -version-info $(LIBSTARPUMPI_INTERFACE_CURRENT):$(LIBSTARPUMPI_INTERFACE_REVISION):$(LIBSTARPUMPI_INTERFACE_AGE) \
  $(MPICC_LDFLAGS) $(FXT_LDFLAGS)
noinst_HEADERS =					\
	starpu_mpi_private.h				\
	starpu_mpi_fxt.h				\
	starpu_mpi_stats.h				\
	starpu_mpi_datatype.h

libstarpumpi_@STARPU_EFFECTIVE_VERSION@_la_SOURCES =	\
	starpu_mpi.c					\
	starpu_mpi_helper.c				\
	starpu_mpi_datatype.c				\
	starpu_mpi_insert_task.c			\
	starpu_mpi_collective.c				\
	starpu_mpi_stats.c


showcheck:
	-cat /dev/null
