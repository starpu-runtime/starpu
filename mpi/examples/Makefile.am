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

if STARPU_MPI_CHECK
if STARPU_HAVE_AM111
LOG_COMPILER	 	=	$(MPIEXEC) -np 2
else
TESTS_ENVIRONMENT 	=	$(MPIEXEC) -np 2
endif
TESTS			=	$(check_PROGRAMS)
endif

check_PROGRAMS =

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo

EXTRA_DIST = 					\
	mpi_lu/mpi_lu-float.h		\
	mpi_lu/mpi_lu-double.h		\
	mpi_lu/plu_example.c		\
	mpi_lu/plu_solve.c		\
	mpi_lu/pxlu.h			\
	mpi_lu/pxlu.c			\
	mpi_lu/pxlu_kernels.h		\
	mpi_lu/pxlu_kernels.c		\
	cholesky/mpi_cholesky.h	\
	cholesky/mpi_cholesky_models.h \
	cholesky/mpi_cholesky_codelets.h \
	../tests/helper.h

examplebindir = $(libdir)/starpu/mpi

examplebin_PROGRAMS =

if STARPU_USE_CUDA
# TODO define NVCCFLAGS
NVCC ?= nvcc

NVCCFLAGS += -I$(top_srcdir)/include/ -I$(top_builddir)/include

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS)

.cu.o:
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I$(top_srcdir)/include/  -I$(top_builddir)/include/
endif

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS)
LIBS = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) 

###################
# Stencil example #
###################
if BUILD_EXAMPLES
examplebin_PROGRAMS +=				\
	stencil/stencil5

stencil_stencil5_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

check_PROGRAMS	+=	\
	stencil/stencil5

##################
# MPI LU example #
##################

if !NO_BLAS_LIB

examplebin_PROGRAMS += 			\
	mpi_lu/plu_example_float	\
	mpi_lu/plu_example_double

mpi_lu_plu_example_float_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS)

mpi_lu_plu_example_float_SOURCES =	\
	mpi_lu/plu_example_float.c	\
	mpi_lu/plu_solve_float.c	\
	mpi_lu/pslu_kernels.c		\
	mpi_lu/pslu.c			\
	$(top_srcdir)/examples/common/blas.c

mpi_lu_plu_example_double_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS)

mpi_lu_plu_example_double_SOURCES =	\
	mpi_lu/plu_example_double.c	\
	mpi_lu/plu_solve_double.c  	\
	mpi_lu/pdlu_kernels.c	    	\
	mpi_lu/pdlu.c		    	\
	$(top_srcdir)/examples/common/blas.c
endif

########################
# MPI Cholesky example #
########################

if !NO_BLAS_LIB
examplebin_PROGRAMS +=		\
	cholesky/mpi_cholesky			\
	cholesky/mpi_cholesky_distributed

cholesky_mpi_cholesky_SOURCES	=		\
	cholesky/mpi_cholesky.c		\
	cholesky/mpi_cholesky_models.c		\
	cholesky/mpi_cholesky_kernels.c	\
	cholesky/mpi_cholesky_codelets.c	\
	$(top_srcdir)/examples/common/blas.c

cholesky_mpi_cholesky_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_BLAS_LDFLAGS)

cholesky_mpi_cholesky_distributed_SOURCES =	\
	cholesky/mpi_cholesky_distributed.c	\
	cholesky/mpi_cholesky_models.c		\
	cholesky/mpi_cholesky_kernels.c	\
	cholesky/mpi_cholesky_codelets.c	\
	$(top_srcdir)/examples/common/blas.c

cholesky_mpi_cholesky_distributed_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_BLAS_LDFLAGS)

check_PROGRAMS +=					\
	cholesky/mpi_cholesky			\
	cholesky/mpi_cholesky_distributed
endif

########################
# Scatter Gather       #
########################

examplebin_PROGRAMS +=		\
	scatter_gather/mpi_scatter_gather

scatter_gather_mpi_scatter_gather_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

check_PROGRAMS +=		\
	scatter_gather/mpi_scatter_gather

###################
# Reduction       #
###################

examplebin_PROGRAMS +=		\
	reduction/mpi_reduction

reduction_mpi_reduction_SOURCES =		\
	reduction/mpi_reduction.c		\
	reduction/mpi_reduction_kernels.c

reduction_mpi_reduction_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

check_PROGRAMS +=		\
	reduction/mpi_reduction

###################
# complex example #
###################

examplebin_PROGRAMS +=				\
	complex/mpi_complex

complex_mpi_complex_SOURCES =		\
	complex/mpi_complex.c		\
	$(top_srcdir)/examples/interface/complex_interface.c

complex_mpi_complex_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

check_PROGRAMS	+=	\
	complex/mpi_complex
endif


showcheck:
	-cat $(TEST_LOGS) /dev/null
