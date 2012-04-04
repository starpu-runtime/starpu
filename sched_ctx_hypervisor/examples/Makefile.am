# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011,2012 Institut National de Recherche en Informatique et Automatique
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
AUTOMAKE_OPTIONS = subdir-objects

if !NO_BLAS_LIB
noinst_PROGRAMS =				\
	cholesky/cholesky_implicit

noinst_HEADERS = 				\
	cholesky/cholesky.h			\
	sched_ctx_utils/sched_ctx_utils.h
endif

AM_LDFLAGS = $(top_builddir)/src/libstarpu-1.0.la 

LIBS = $(top_srcdir)/sched_ctx_hypervisor/src/libsched_ctx_hypervisor.la

AM_CPPFLAGS =						\
  -I$(top_srcdir)/include				\
  -I$(top_srcdir)/examples				\
  -I$(top_builddir)/include				\
  -I$(top_srcdir)/sched_ctx_hypervisor/include		\
  $(STARPU_OPENCL_CPPFLAGS) $(STARPU_CUDA_CPPFLAGS)


if !NO_BLAS_LIB

cholesky_cholesky_implicit_SOURCES =		\
	cholesky/cholesky_implicit.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	sched_ctx_utils/sched_ctx_utils.c	\
	$(top_srcdir)/examples/common/blas.c

cholesky_cholesky_implicit_LDADD =		\
	$(top_srcdir)/sched_ctx_hypervisor/src/libsched_ctx_hypervisor.la
	$(STARPU_BLAS_LDFLAGS)

endif