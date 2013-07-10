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
TESTS			=	$(starpu_mpi_TESTS)
endif

check_PROGRAMS = $(LOADER) $(starpu_mpi_TESTS)

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo

EXTRA_DIST = 					\
	user_defined_datatype_value.h

examplebindir = $(libdir)/starpu/examples/mpi

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
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include -I$(top_srcdir)/src -I$(top_builddir)/src -I$(top_srcdir)/examples/
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(FXT_LDFLAGS)

########################
# Unit testcases       #
########################

starpu_mpi_TESTS =				\
	datatypes				\
	pingpong				\
	mpi_test				\
	mpi_isend				\
	mpi_earlyrecv				\
	mpi_earlyrecv2				\
	mpi_irecv				\
	mpi_isend_detached			\
	mpi_irecv_detached			\
	mpi_detached_tag			\
	mpi_redux				\
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
	multiple_send				\
	mpi_scatter_gather			\
	mpi_reduction				\
	user_defined_datatype

noinst_PROGRAMS =				\
	datatypes				\
	pingpong				\
	mpi_test				\
	mpi_isend				\
	mpi_earlyrecv				\
	mpi_earlyrecv2				\
	mpi_irecv				\
	mpi_isend_detached			\
	mpi_irecv_detached			\
	mpi_detached_tag			\
	mpi_redux				\
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
	multiple_send				\
	mpi_scatter_gather			\
	mpi_reduction				\
	user_defined_datatype

mpi_isend_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_earlyrecv_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_earlyrecv2_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_irecv_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_isend_detached_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_irecv_detached_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_detached_tag_LDADD =				\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_redux_LDADD =					\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
datatypes_LDADD =					\
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
mpi_scatter_gather_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
mpi_reduction_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
user_defined_datatype_LDADD =			\
	../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la

ring_SOURCES = ring.c
ring_async_SOURCES = ring_async.c
ring_async_implicit_SOURCES = ring_async_implicit.c
if STARPU_USE_CUDA
ring_SOURCES += ring_kernel.cu
ring_async_SOURCES += ring_kernel.cu
ring_async_implicit_SOURCES += ring_kernel.cu
endif
mpi_reduction_SOURCES = mpi_reduction.c
mpi_reduction_SOURCES += mpi_reduction_kernels.c
user_defined_datatype_SOURCES = user_defined_datatype.c
user_defined_datatype_SOURCES += $(top_srcdir)/examples/interface/complex_interface.c

mpi_earlyrecv2_SOURCES = mpi_earlyrecv2.c
mpi_earlyrecv2_SOURCES += $(top_srcdir)/examples/interface/complex_interface.c

showcheck:
	-cat $(TEST_LOGS) /dev/null
