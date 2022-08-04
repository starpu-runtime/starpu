# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

include $(top_srcdir)/starpu-tests.mk
include $(top_srcdir)/starpu-loader.mk

LAUNCHER_ENV += OMP_NUM_THREADS=4

TESTS 	= $(STARPU_OPENMP_LLVM_EXAMPLES)
STARPU_OPENMP_LLVM_EXAMPLES =

# we want to compile the application just with clang --fopenmp
CC		=  $(PROG_CLANG)
AM_CPPFLAGS	=
AM_CFLAGS	+= -Wall -g -fopenmp
# ideally, we should create a link libomp.so.5 to
# libstarpu_openmp_llvm-@STARPU_EFFECTIVE_VERSION@.so but because it
# is libtool we use LDADD
LDADD		 = $(top_builddir)/starpu_openmp_llvm/src/libstarpu_openmp_llvm-@STARPU_EFFECTIVE_VERSION@.la

examplebindir = $(libdir)/starpu/examples/starpu_openmp_llvm
examplebin_PROGRAMS = $(STARPU_OPENMP_LLVM_EXAMPLES)
check_PROGRAMS = $(LOADER) $(STARPU_OPENMP_LLVM_EXAMPLES)

STARPU_OPENMP_LLVM_EXAMPLES += hello-task

exampledir = $(libdir)/starpu/examples/starpu_openmp_llvm
example_DATA = README hello-task.c

EXTRA_DIST = README
