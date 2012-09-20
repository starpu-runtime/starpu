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

examplebindir = $(libdir)/starpu/examples/mpi

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
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/mpi/ -I$(top_srcdir)/src/  -I$(top_builddir)/src -I$(top_builddir)/include -I$(top_srcdir)/mpi/include -I$(top_srcdir)/mpi/src
AM_LDFLAGS = $(STARPU_CUDA_LDFLAGS) $(STARPU_OPENCL_LDFLAGS)

########################
# Unit testcases       #
########################

check_PROGRAMS +=				\
	pingpong				\
	mpi_test				\
	mpi_isend				\
	mpi_irecv				\
	mpi_isend_detached			\
	mpi_irecv_detached			\
	mpi_detached_tag			\
	ring					\
	ring_async				\
	ring_async_implicit			\
	block_interface				\
	block_interface_pinned			\
	insert_task				\
	insert_task_cache			\
	insert_task_block			\
	insert_task_owner			\
	insert_task_owner2			\
	insert_task_owner_data			\
	multiple_send

noinst_PROGRAMS =				\
	pingpong				\
	mpi_test				\
	mpi_isend				\
	mpi_irecv				\
	mpi_isend_detached			\
	mpi_irecv_detached			\
	mpi_detached_tag			\
	ring					\
	ring_async				\
	ring_async_implicit			\
	block_interface				\
	block_interface_pinned			\
	insert_task				\
	insert_task_cache			\
	insert_task_block			\
	insert_task_owner			\
	insert_task_owner2			\
	insert_task_owner_data			\
	multiple_send

mpi_isend_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_irecv_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_isend_detached_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_irecv_detached_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_detached_tag_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
pingpong_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_test_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
ring_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
ring_async_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
ring_async_implicit_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
block_interface_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
block_interface_pinned_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
insert_task_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
insert_task_cache_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
insert_task_block_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
insert_task_owner_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
insert_task_owner2_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
insert_task_owner_data_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
multiple_send_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

ring_SOURCES = ring.c
ring_async_SOURCES = ring_async.c
ring_async_implicit_SOURCES = ring_async_implicit.c
if STARPU_USE_CUDA
ring_SOURCES += ring_kernel.cu
ring_async_SOURCES += ring_kernel.cu
ring_async_implicit_SOURCES += ring_kernel.cu
endif

showcheck:
	-cat $(TEST_LOGS) /dev/null
