# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

CLEANFILES = *.gcno *.gcda

AM_CFLAGS += -fPIC
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/include -I$(top_builddir)/src -I$(top_srcdir)/src -I$(top_srcdir)/julia/src $(STARPU_H_CPPFLAGS)
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ `@JULIA@ $(top_srcdir)/julia/src/openblas_ldflags.jl` $(STARPU_EXPORTED_LIBS)

SUBDIRS = dynamic_compiler

lib_LTLIBRARIES = libstarpujulia-@STARPU_EFFECTIVE_VERSION@.la

noinst_HEADERS =

libstarpujulia_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS = $(ldflags) -no-undefined			\
  -version-info $(LIBSTARPUJULIA_INTERFACE_CURRENT):$(LIBSTARPUJULIA_INTERFACE_REVISION):$(LIBSTARPUJULIA_INTERFACE_AGE)

libstarpujulia_@STARPU_EFFECTIVE_VERSION@_la_SOURCES = 						\
	callback_wrapper.c \
	blas_wrapper.c \
	blas.c

EXTRA_DIST = blas.h				\
	blas.jl  				\
	data.jl					\
	destructible.jl				\
	globals.jl				\
	init.jl					\
	linked_list.jl				\
	perfmodel.jl				\
	StarPU.jl				\
	task_dep.jl				\
	task.jl					\
	translate_headers.jl			\
	utils.jl				\
	compiler/c.jl				\
	compiler/cuda.jl			\
	compiler/expression_manipulation.jl	\
	compiler/expressions.jl			\
	compiler/file_generation.jl		\
	compiler/include.jl			\
	compiler/parsing.jl			\
	compiler/utils.jl
