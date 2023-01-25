# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/make/starpu-notests.mk

SUBDIRS =

AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_srcdir)/mpi/include -I$(top_builddir)/src -I$(top_srcdir)/src -DSTARPU_REPLAY_MPI $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(top_builddir)/mpi/src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la $(STARPU_EXPORTED_LIBS)

CC=$(CC_OR_MPICC)
CCLD=$(CC_OR_MPICC)

starpu_replay.c starpu_replay_sched.c:
	$(V_ln) $(LN_S) $(top_srcdir)/tools/$(notdir $@) $@

if STARPU_SIMGRID
bin_PROGRAMS = starpu_replay_mpi

starpu_replay_mpi_SOURCES = \
	starpu_replay.c \
	starpu_replay_sched.c
endif
