# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2013, 2015  Université de Bordeaux
# Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  CNRS
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

if STARPU_QUICK_CHECK
MPI			=	$(MPIEXEC) -np 2
else
MPI			=	$(MPIEXEC) -np 4
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER	 	=	$(MPI) $(LOADER_BIN)
else
TESTS_ENVIRONMENT 	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(MPI) $(LOADER_BIN)
endif

if STARPU_MPI_CHECK
TESTS			=	$(starpu_mpi_EXAMPLES)
endif

check_PROGRAMS = $(LOADER) $(starpu_mpi_EXAMPLES)
starpu_mpi_EXAMPLES =

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log

EXTRA_DIST = 				\
	mpi_lu/mpi_lu-float.h		\
	mpi_lu/mpi_lu-double.h		\
	mpi_lu/plu_example.c		\
	mpi_lu/plu_implicit_example.c	\
	mpi_lu/plu_outofcore_example.c	\
	mpi_lu/plu_solve.c		\
	mpi_lu/pxlu.h			\
	mpi_lu/pxlu.c			\
	mpi_lu/pxlu_implicit.c		\
	mpi_lu/pxlu_kernels.h		\
	mpi_lu/pxlu_kernels.c		\
	matrix_decomposition/mpi_cholesky.h 		\
	matrix_decomposition/mpi_cholesky_codelets.h 	\
	matrix_decomposition/mpi_cholesky_kernels.h	\
	matrix_decomposition/mpi_cholesky_models.h 	\
	matrix_decomposition/mpi_decomposition_params.h	\
	matrix_decomposition/mpi_decomposition_matrix.h	\
	user_datatype/my_interface.h			\
	helper.h

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

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS) -Wno-unused
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(FXT_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

###################
# Stencil example #
###################
if BUILD_EXAMPLES
examplebin_PROGRAMS +=				\
	stencil/stencil5

stencil_stencil5_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la -lm

starpu_mpi_EXAMPLES	+=	\
	stencil/stencil5
endif

##################
# MPI LU example #
##################

if BUILD_EXAMPLES
if !NO_BLAS_LIB

examplebin_PROGRAMS += 			\
	mpi_lu/plu_example_float	\
	mpi_lu/plu_example_double	\
	mpi_lu/plu_implicit_example_float	\
	mpi_lu/plu_implicit_example_double	\
	mpi_lu/plu_outofcore_example_float	\
	mpi_lu/plu_outofcore_example_double

mpi_lu_plu_example_float_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_example_float_SOURCES =	\
	mpi_lu/plu_example_float.c	\
	mpi_lu/plu_solve_float.c	\
	mpi_lu/pslu_kernels.c		\
	mpi_lu/pslu.c			\
	../../examples/common/blas.c

mpi_lu_plu_example_double_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_example_double_SOURCES =	\
	mpi_lu/plu_example_double.c	\
	mpi_lu/plu_solve_double.c  	\
	mpi_lu/pdlu_kernels.c	    	\
	mpi_lu/pdlu.c		    	\
	../../examples/common/blas.c

mpi_lu_plu_implicit_example_float_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_implicit_example_float_SOURCES =	\
	mpi_lu/plu_implicit_example_float.c	\
	mpi_lu/plu_solve_float.c		\
	mpi_lu/pslu_kernels.c			\
	mpi_lu/pslu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_implicit_example_double_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_implicit_example_double_SOURCES =	\
	mpi_lu/plu_implicit_example_double.c	\
	mpi_lu/plu_solve_double.c		\
	mpi_lu/pdlu_kernels.c			\
	mpi_lu/pdlu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_outofcore_example_float_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_outofcore_example_float_SOURCES =	\
	mpi_lu/plu_outofcore_example_float.c	\
	mpi_lu/plu_solve_float.c		\
	mpi_lu/pslu_kernels.c			\
	mpi_lu/pslu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_outofcore_example_double_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_outofcore_example_double_SOURCES =	\
	mpi_lu/plu_outofcore_example_double.c	\
	mpi_lu/plu_solve_double.c		\
	mpi_lu/pdlu_kernels.c			\
	mpi_lu/pdlu_implicit.c			\
	../../examples/common/blas.c
endif
endif

########################
# MPI Cholesky example #
########################

if BUILD_EXAMPLES
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
	../../examples/common/blas.c

matrix_decomposition_mpi_cholesky_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_BLAS_LDFLAGS) -lm

matrix_decomposition_mpi_cholesky_distributed_SOURCES =	\
	matrix_decomposition/mpi_cholesky_distributed.c	\
	matrix_decomposition/mpi_cholesky_models.c		\
	matrix_decomposition/mpi_cholesky_kernels.c	\
	matrix_decomposition/mpi_cholesky_codelets.c	\
	matrix_decomposition/mpi_decomposition_params.c	\
	matrix_decomposition/mpi_decomposition_matrix.c	\
	../../examples/common/blas.c

matrix_decomposition_mpi_cholesky_distributed_LDADD =	\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la	\
	$(STARPU_BLAS_LDFLAGS) -lm

starpu_mpi_EXAMPLES +=				\
	matrix_decomposition/mpi_cholesky			\
	matrix_decomposition/mpi_cholesky_distributed
endif
endif

###################
# complex example #
###################

if BUILD_EXAMPLES
examplebin_PROGRAMS +=			\
	complex/mpi_complex

complex_mpi_complex_SOURCES =		\
	complex/mpi_complex.c		\
	../../examples/interface/complex_interface.c

complex_mpi_complex_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

starpu_mpi_EXAMPLES	+=			\
	complex/mpi_complex
endif

#########################
# user_datatype example #
#########################

if BUILD_EXAMPLES
examplebin_PROGRAMS +=				\
	user_datatype/user_datatype

user_datatype_user_datatype_SOURCES =		\
	user_datatype/user_datatype.c		\
	user_datatype/my_interface.c

user_datatype_user_datatype_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

starpu_mpi_EXAMPLES	+=			\
	user_datatype/user_datatype
endif

###################
# comm example #
###################

if BUILD_EXAMPLES
examplebin_PROGRAMS +=			\
	comm/comm			\
	comm/mix_comm

comm_comm_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
comm_mix_comm_LDADD =		\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

starpu_mpi_EXAMPLES	+=			\
	comm/comm				\
	comm/mix_comm
endif


showcheck:
	-cat $(TEST_LOGS) /dev/null
	! grep -q "ERROR: AddressSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "WARNING: AddressSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "ERROR: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "WARNING: ThreadSanitizer: " $(TEST_LOGS) /dev/null
