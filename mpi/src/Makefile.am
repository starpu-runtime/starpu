# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

include $(top_srcdir)/starpu-notests.mk

CC=$(MPICC)
CCLD=$(MPICC)

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS) $(NMAD_CFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS) $(NMAD_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/src/ -I$(top_builddir)/src -I$(top_builddir)/include -I$(top_srcdir)/mpi/include -I$(top_srcdir)/mpi/src
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS) $(NMAD_LDFLAGS)

ldflags =

if STARPU_HAVE_WINDOWS

LC_MESSAGES=C
export LC_MESSAGES

ldflags += -Xlinker --output-def -Xlinker .libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.def

if STARPU_HAVE_MS_LIB
.libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.lib: libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la dolib
	./dolib "$(STARPU_MS_LIB)" $(STARPU_MS_LIB_ARCH) .libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.def @STARPU_EFFECTIVE_VERSION@ $(libstarpumpi_so_version) .libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.lib
all-local: .libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.lib
endif STARPU_HAVE_MS_LIB

install-exec-hook:
	$(INSTALL) .libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.def $(DESTDIR)$(libdir)
if STARPU_HAVE_MS_LIB
	$(INSTALL) .libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.lib $(DESTDIR)$(libdir)
	$(INSTALL) .libs/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.exp $(DESTDIR)$(libdir)
endif STARPU_HAVE_MS_LIB

endif STARPU_HAVE_WINDOWS

lib_LTLIBRARIES = libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

libstarpumpi_@STARPU_EFFECTIVE_VERSION@_la_LIBADD = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la
libstarpumpi_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS = $(ldflags) -no-undefined					\
  -version-info $(LIBSTARPUMPI_INTERFACE_CURRENT):$(LIBSTARPUMPI_INTERFACE_REVISION):$(LIBSTARPUMPI_INTERFACE_AGE) \
  $(MPICC_LDFLAGS) $(FXT_LDFLAGS)
noinst_HEADERS =					\
	starpu_mpi_private.h				\
	starpu_mpi_fxt.h				\
	starpu_mpi_stats.h				\
	starpu_mpi_datatype.h				\
	starpu_mpi_cache.h				\
	starpu_mpi_select_node.h			\
	starpu_mpi_cache_stats.h			\
	starpu_mpi_task_insert.h			\
	starpu_mpi_init.h				\
	mpi/starpu_mpi_early_data.h			\
	mpi/starpu_mpi_early_request.h			\
	mpi/starpu_mpi_sync_data.h			\
	mpi/starpu_mpi_comm.h				\
	mpi/starpu_mpi_tag.h				\
	mpi/starpu_mpi_driver.h				\
	mpi/starpu_mpi_mpi_backend.h			\
	nmad/starpu_mpi_nmad_backend.h			\
	load_balancer/policy/data_movements_interface.h	\
	load_balancer/policy/load_data_interface.h	\
	load_balancer/policy/load_balancer_policy.h

libstarpumpi_@STARPU_EFFECTIVE_VERSION@_la_SOURCES =	\
	starpu_mpi.c					\
	starpu_mpi_req.c				\
	starpu_mpi_coop_sends.c				\
	starpu_mpi_helper.c				\
	starpu_mpi_datatype.c				\
	starpu_mpi_task_insert.c			\
	starpu_mpi_collective.c				\
	starpu_mpi_stats.c				\
	starpu_mpi_private.c				\
	starpu_mpi_cache.c				\
	starpu_mpi_select_node.c			\
	starpu_mpi_cache_stats.c			\
	starpu_mpi_fortran.c				\
	starpu_mpi_task_insert_fortran.c		\
	starpu_mpi_init.c				\
	nmad/starpu_mpi_nmad.c				\
	nmad/starpu_mpi_nmad_backend.c			\
	mpi/starpu_mpi_mpi.c				\
	mpi/starpu_mpi_mpi_backend.c			\
	mpi/starpu_mpi_early_data.c			\
	mpi/starpu_mpi_early_request.c			\
	mpi/starpu_mpi_sync_data.c			\
	mpi/starpu_mpi_comm.c				\
	mpi/starpu_mpi_tag.c				\
	load_balancer/policy/data_movements_interface.c	\
	load_balancer/policy/load_data_interface.c	\
	load_balancer/policy/load_heat_propagation.c	\
	load_balancer/load_balancer.c
