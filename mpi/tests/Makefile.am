# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2013       Thibaut Lambert
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

CC=$(MPICC)
CCLD=$(MPICC)

noinst_PROGRAMS		=

if STARPU_HAVE_WINDOWS
LOADER_BIN		=
else
loader_CPPFLAGS 	= 	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
if !STARPU_SIMGRID
LOADER			=	loader
LOADER_BIN		=	$(abs_top_builddir)/mpi/tests/$(LOADER)
noinst_PROGRAMS		+=	loader
endif
loader_SOURCES		=	../../tests/loader.c
endif

if STARPU_SIMGRID
MPI			=	$(abs_top_builddir)/tools/starpu_smpirun -np 4 -platform $(abs_top_srcdir)/tools/perfmodels/cluster.xml -hostfile $(abs_top_srcdir)/tools/perfmodels/hostfile
else
# we always test on 4 processes, the execution time is not that bigger
if STARPU_QUICK_CHECK
MPI			=	$(MPIEXEC) $(MPIEXEC_ARGS) -np 4
else
MPI			=	$(MPIEXEC) $(MPIEXEC_ARGS) -np 4
endif
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	STARPU_WORKERS_NOBIND=1 STARPU_NCPU=3 top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER	 	=	$(MPI) $(LOADER_BIN)
else
TESTS_ENVIRONMENT 	=	STARPU_WORKERS_NOBIND=1 STARPU_NCPU=3 top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(MPI) $(LOADER_BIN)
endif

if STARPU_MPI_CHECK
TESTS			=	$(starpu_mpi_TESTS)
endif

check_PROGRAMS = $(LOADER) $(starpu_mpi_TESTS)

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log

EXTRA_DIST = 					\
	user_defined_datatype_value.h		\
	helper.h

examplebindir = $(libdir)/starpu/examples/mpi

examplebin_PROGRAMS =

if STARPU_USE_CUDA
if STARPU_COVERITY
include $(top_srcdir)/starpu-mynvcc.mk
else
NVCCFLAGS += --compiler-options -fno-strict-aliasing  -I$(top_srcdir)/include/ -I$(top_builddir)/include/ $(HWLOC_CFLAGS)

.cu.cubin:
	$(V_nvcc) $(NVCC) -cubin $< -o $@ $(NVCCFLAGS)

.cu.o:
	$(V_nvcc) $(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif
endif

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS) -Wno-unused
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ ../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include -I$(top_srcdir)/mpi/src -I$(top_srcdir)/src -I$(top_builddir)/src -I$(top_srcdir)/examples/
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(FXT_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

########################
# Unit testcases       #
########################

starpu_mpi_TESTS =

starpu_mpi_TESTS +=				\
	broadcast				\
	callback				\
	driver					\
	early_request				\
	gather					\
	gather2					\
	insert_task				\
	insert_task_block			\
	insert_task_count			\
	insert_task_dyn_handles			\
	insert_task_node_choice			\
	insert_task_owner			\
	insert_task_owner2			\
	insert_task_owner_data			\
	matrix					\
	matrix2					\
	mpi_detached_tag			\
	mpi_earlyrecv				\
	mpi_irecv				\
	mpi_irecv_detached			\
	mpi_isend				\
	mpi_isend_detached			\
	mpi_reduction				\
	mpi_redux				\
	mpi_scatter_gather			\
	mpi_test				\
	multiple_send				\
	pingpong				\
	policy_register				\
	policy_register_many			\
	policy_selection			\
	policy_selection2			\
	ring					\
	ring_async				\
	ring_async_implicit			\
	ring_sync				\
	ring_sync_detached			\
	temporary				\
	user_defined_datatype			\
	early_stuff

if !STARPU_SIMGRID
# missing support in simgrid
starpu_mpi_TESTS +=				\
	attr					\
	display_bindings			\
	mpi_earlyrecv2				\
	mpi_earlyrecv2_sync			\
	block_interface				\
	block_interface_pinned			\
	insert_task_compute			\
	insert_task_sent_cache			\
	insert_task_recv_cache			\
	insert_task_seq				\
	tags_checking				\
	sync

if STARPU_USE_MPI_MPI
starpu_mpi_TESTS +=				\
	load_balancer
endif

# Expected to fail
starpu_mpi_TESTS +=				\
	policy_register_toomany			\
	policy_unregister			\
	starpu_redefine
endif

noinst_PROGRAMS +=				\
	datatypes				\
	pingpong				\
	mpi_test				\
	mpi_isend				\
	mpi_earlyrecv				\
	mpi_earlyrecv2				\
	mpi_earlyrecv2_sync			\
	mpi_irecv				\
	mpi_isend_detached			\
	mpi_irecv_detached			\
	mpi_detached_tag			\
	mpi_redux				\
	ring					\
	ring_sync				\
	ring_sync_detached			\
	ring_async				\
	ring_async_implicit			\
	temporary				\
	early_stuff				\
	block_interface				\
	block_interface_pinned			\
	attr					\
	broadcast				\
	callback				\
	matrix					\
	matrix2					\
	insert_task				\
	insert_task_compute			\
	insert_task_sent_cache			\
	insert_task_recv_cache			\
	insert_task_block			\
	insert_task_owner			\
	insert_task_owner2			\
	insert_task_owner_data			\
	insert_task_node_choice			\
	insert_task_count			\
	insert_task_dyn_handles			\
	insert_task_seq				\
	multiple_send				\
	mpi_scatter_gather			\
	mpi_reduction				\
	user_defined_datatype			\
	tags_checking				\
	sync					\
	gather					\
	gather2					\
	policy_register				\
	policy_register_many			\
	policy_register_toomany			\
	policy_unregister			\
	policy_selection			\
	policy_selection2			\
	early_request				\
	starpu_redefine				\
	load_balancer				\
	driver					\
	display_bindings


XFAIL_TESTS=					\
	policy_register_toomany			\
	policy_unregister			\
	starpu_redefine

ring_SOURCES = ring.c
ring_sync_SOURCES = ring_sync.c
ring_sync_detached_SOURCES = ring_sync_detached.c
ring_async_SOURCES = ring_async.c
ring_async_implicit_SOURCES = ring_async_implicit.c
insert_task_count_SOURCES = insert_task_count.c
if STARPU_USE_CUDA
ring_SOURCES += ring_kernel.cu
ring_sync_SOURCES += ring_kernel.cu
ring_sync_detached_SOURCES += ring_kernel.cu
ring_async_SOURCES += ring_kernel.cu
ring_async_implicit_SOURCES += ring_kernel.cu
insert_task_count_SOURCES += ring_kernel.cu
endif
mpi_reduction_SOURCES = mpi_reduction.c
mpi_reduction_SOURCES += mpi_reduction_kernels.c
user_defined_datatype_SOURCES = user_defined_datatype.c
user_defined_datatype_SOURCES += ../../examples/interface/complex_interface.c

mpi_earlyrecv2_SOURCES = mpi_earlyrecv2.c
mpi_earlyrecv2_SOURCES += ../../examples/interface/complex_interface.c
mpi_earlyrecv2_sync_SOURCES = mpi_earlyrecv2_sync.c
mpi_earlyrecv2_sync_SOURCES += ../../examples/interface/complex_interface.c
