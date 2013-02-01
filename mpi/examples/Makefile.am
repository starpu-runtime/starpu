# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2012  Université de Bordeaux 1
# Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

if STARPU_HAVE_WINDOWS
LOADER_BIN		=
else
loader_CPPFLAGS 	= 	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER			=	loader
LOADER_BIN		=	$(abs_top_builddir)/mpi/tests/$(LOADER)
loader_SOURCES		=	../../tests/loader.c
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER	 	=	$(MPIEXEC) -np 2 $(LOADER_BIN)
else
TESTS_ENVIRONMENT 	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(MPIEXEC) -np 4
endif

if STARPU_MPI_CHECK
TESTS			=	$(starpu_mpi_EXAMPLES)
endif

check_PROGRAMS = $(LOADER) $(starpu_mpi_EXAMPLES)
starpu_mpi_EXAMPLES =

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
	matrix_decomposition/mpi_cholesky_codelets.h 	\
	matrix_decomposition/mpi_cholesky_kernels.h	\
	matrix_decomposition/mpi_cholesky_models.h 	\
	matrix_decomposition/mpi_decomposition_params.h	\
	matrix_decomposition/mpi_decomposition_matrix.h	\
	../tests/helper.h

examplebindir = $(libdir)/starpu/mpi

examplebin_PROGRAMS =

if STARPU_USE_CUDA
NVCCFLAGS += --compiler-options -fno-strict-aliasing  -I$(top_srcdir)/include/ -I$(top_builddir)/include/ $(HWLOC_CFLAGS)

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ $(NVCCFLAGS)

.cu.o:
	$(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(FXT_LDFLAGS)

###################
# Stencil example #
###################
if BUILD_EXAMPLES
examplebin_PROGRAMS +=				\
	stencil/stencil5

stencil_stencil5_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

starpu_mpi_EXAMPLES	+=	\
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
	matrix_decomposition/mpi_cholesky			\
	matrix_decomposition/mpi_cholesky_distributed

matrix_decomposition_mpi_cholesky_SOURCES	=		\
	matrix_decomposition/mpi_cholesky.c		\
	matrix_decomposition/mpi_cholesky_models.c		\
	matrix_decomposition/mpi_cholesky_kernels.c	\
	matrix_decomposition/mpi_cholesky_codelets.c	\
	matrix_decomposition/mpi_decomposition_params.c	\
	matrix_decomposition/mpi_decomposition_matrix.c	\
	$(top_srcdir)/examples/common/blas.c

matrix_decomposition_mpi_cholesky_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_BLAS_LDFLAGS)

matrix_decomposition_mpi_cholesky_distributed_SOURCES =	\
	matrix_decomposition/mpi_cholesky_distributed.c	\
	matrix_decomposition/mpi_cholesky_models.c		\
	matrix_decomposition/mpi_cholesky_kernels.c	\
	matrix_decomposition/mpi_cholesky_codelets.c	\
	matrix_decomposition/mpi_decomposition_params.c	\
	matrix_decomposition/mpi_decomposition_matrix.c	\
	$(top_srcdir)/examples/common/blas.c

matrix_decomposition_mpi_cholesky_distributed_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_BLAS_LDFLAGS)

starpu_mpi_EXAMPLES +=				\
	matrix_decomposition/mpi_cholesky			\
	matrix_decomposition/mpi_cholesky_distributed
endif

###################
# complex example #
###################

examplebin_PROGRAMS +=			\
	complex/mpi_complex

complex_mpi_complex_SOURCES =		\
	complex/mpi_complex.c		\
	$(top_srcdir)/examples/interface/complex_interface.c

complex_mpi_complex_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

starpu_mpi_EXAMPLES	+=			\
	complex/mpi_complex
endif


showcheck:
	-cat $(TEST_LOGS) /dev/null
