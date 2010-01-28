#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

CC=$(MPICC)

LIBS = $(top_builddir)/src/libstarpu.la $(top_builddir)/mpi/libstarpumpi.la @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include  -I$(top_srcdir)/mpi/

TESTS = $(check_PROGRAMS)

check_PROGRAMS =

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo

if USE_CUDA

# TODO define NVCCFLAGS
NVCC ?= nvcc

.cu.o:
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I$(top_srcdir)/include/ -I$(top_builddir)/include/
endif

examplebindir = $(libdir)/starpu/mpi/examples/

examplebin_PROGRAMS =

##################
# MPI LU example #
##################

if !NO_BLAS_LIB

examplebin_PROGRAMS += 				\
	mpi_lu/plu_example_float		\
	mpi_lu/plu_example_double

mpi_lu_plu_example_float_SOURCES =		\
	mpi_lu/plu_example_float.c		\
	mpi_lu/plu_solve_float.c		\
	mpi_lu/pslu_kernels.c			\
	mpi_lu/pslu.c				\
	$(top_srcdir)/examples/common/blas.c

mpi_lu_plu_example_double_SOURCES =		\
	mpi_lu/plu_example_double.c		\
	mpi_lu/plu_solve_double.c		\
	mpi_lu/pdlu_kernels.c			\
	mpi_lu/pdlu.c				\
	$(top_srcdir)/examples/common/blas.c
endif
