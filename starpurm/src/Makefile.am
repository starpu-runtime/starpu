# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

CLEANFILES = *.gcno *.gcda *.linkinfo

AM_CPPFLAGS	= -I$(top_srcdir)/include -I$(top_srcdir)/src -I$(top_builddir)/src -I$(top_builddir)/include
AM_CPPFLAGS	+= -I$(top_srcdir)/starpurm/include -I$(top_srcdir)/starpurm/src -I$(top_builddir)/starpurm/src -I$(top_builddir)/starpurm/include $(STARPU_H_CPPFLAGS)
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS) $(HWLOC_LIBS)

libstarpurm_so_version = $(LIBSTARPURM_INTERFACE_CURRENT):$(LIBSTARPURM_INTERFACE_REVISION):$(LIBSTARPURM_INTERFACE_AGE)

lib_LTLIBRARIES = libstarpurm-@STARPU_EFFECTIVE_VERSION@.la

libstarpurm_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS	= $(ldflags) -no-undefined -version-info $(libstarpurm_so_version)
libstarpurm_@STARPU_EFFECTIVE_VERSION@_la_SOURCES	= \
	starpurm.c

if STARPURM_HAVE_DLB
libstarpurm_@STARPU_EFFECTIVE_VERSION@_la_SOURCES	+= \
	starpurm_dlb.c
endif

noinst_HEADERS = 					\
	starpurm_private.h
