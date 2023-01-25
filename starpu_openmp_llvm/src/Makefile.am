# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2019-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS += $(FXT_CFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/src/ -I$(top_builddir)/src -I$(top_builddir)/include $(STARPU_H_CPPFLAGS)
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(FXT_LDFLAGS) $(FXT_LIBS)

libstarpu_openmp_llvm_so_version = $(LIBSTARPU_OPENMP_LLVM_INTERFACE_CURRENT):$(LIBSTARPU_OPENMP_LLVM_INTERFACE_REVISION):$(LIBSTARPU_OPENMP_LLVM_INTERFACE_AGE)

lib_LTLIBRARIES = libstarpu_openmp_llvm-@STARPU_EFFECTIVE_VERSION@.la

libstarpu_openmp_llvm_@STARPU_EFFECTIVE_VERSION@_la_CPPFLAGS	= $(AM_CPPFLAGS)
libstarpu_openmp_llvm_@STARPU_EFFECTIVE_VERSION@_la_CFLAGS	= $(AM_CFLAGS)
libstarpu_openmp_llvm_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS	= $(AM_LDFLAGS) -no-undefined -version-info $(libstarpu_openmp_llvm_so_version)
libstarpu_openmp_llvm_@STARPU_EFFECTIVE_VERSION@_la_LIBADD	= $(AM_LIBADD) $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la $(HWLOC_LIBS)
libstarpu_openmp_llvm_@STARPU_EFFECTIVE_VERSION@_la_SOURCES	= \
	openmp_runtime_support_llvm.c

