# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique
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

noinst_PROGRAMS =				\
  matrix-mult stencil5

if !NO_BLAS_LIB
noinst_PROGRAMS +=				\
  cholesky/cholesky
endif

AM_LDFLAGS = $(top_builddir)/src/libstarpu.la

AM_CPPFLAGS =						\
  -I$(top_srcdir)/include				\
  -I$(top_srcdir)/examples				\
  $(STARPU_OPENCL_CPPFLAGS) $(STARPU_CUDA_CPPFLAGS)

AM_CFLAGS = -fplugin="$(builddir)/../src/.libs/starpu.so" -Wall

noinst_HEADERS =				\
  cholesky/cholesky.h				\
  cholesky/cholesky_kernels.h

if !NO_BLAS_LIB
cholesky_cholesky_SOURCES	=		\
	cholesky/cholesky.c		\
	cholesky/cholesky_models.c	\
	cholesky/cholesky_kernels.c	\
	$(top_srcdir)/examples/common/blas.c

cholesky_cholesky_LDADD	=	\
	$(STARPU_BLAS_LDFLAGS)
endif

