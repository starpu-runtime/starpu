# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(top_builddir)/sc_hypervisor/src/libsc_hypervisor.la
AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/examples -I$(top_builddir)/include -I$(top_srcdir)/sc_hypervisor/include -I$(top_srcdir)/sc_hypervisor/examples
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_GLPK_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

noinst_PROGRAMS =				\
	app_driven_test/app_driven_test		\
	lp_test/lp_test				\
	lp_test/lp_resize_test			\
	hierarchical_ctxs/resize_hierarchical_ctxs

if !STARPU_NO_BLAS_LIB
noinst_PROGRAMS +=				\
	cholesky/cholesky_implicit

noinst_HEADERS = 				\
	cholesky/cholesky.h			\
	sched_ctx_utils/sched_ctx_utils.h
endif

if !STARPU_NO_BLAS_LIB

cholesky_cholesky_implicit_SOURCES =		\
	cholesky/cholesky_implicit.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	sched_ctx_utils/sched_ctx_utils.c	\
	../../examples/common/blas.c

cholesky_cholesky_implicit_LDADD =		\
	$(top_builddir)/sc_hypervisor/src/libsc_hypervisor.la \
	$(STARPU_BLAS_LDFLAGS)

endif

app_driven_test_app_driven_test_SOURCES =		\
	app_driven_test/app_driven_test.c

app_driven_test_app_driven_test_LDADD =		\
	$(top_builddir)/sc_hypervisor/src/libsc_hypervisor.la
