# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2013       Joris Pablo
# Copyright (C) 2013       Thibaut Lambert
# Copyright (C) 2017       Erwan Leria
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
include $(top_srcdir)/starpu.mk

SUBDIRS =

AM_CFLAGS = $(HWLOC_CFLAGS) $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(STARPU_COI_CPPFLAGS) $(GLOBAL_AM_CFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(FXT_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_builddir)/src -I$(top_srcdir)/src -DSTARPU_REPLAY_MPI
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@ $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

CC=$(CC_OR_MPICC)
CCLD=$(CC_OR_MPICC)

if STARPU_USE_MPI
LIBS += $(top_builddir)/mpi/src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
AM_CPPFLAGS += -I$(top_srcdir)/mpi/include
endif

bin_PROGRAMS = starpu_replay_mpi

starpu_replay.c starpu_replay_sched.c:
	$(LN_S) $(top_srcdir)/tools/$(notdir $@) $@

starpu_replay_mpi_SOURCES = \
	starpu_replay.c \
	starpu_replay_sched.c
