# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/make/starpu-tests.mk

SUFFIXES = .hip

CC=$(MPICC)
CCLD=$(MPICC)

include $(top_srcdir)/make/starpu-loader.mk
if STARPU_SIMGRID
LOADER_BIN = $(LAUNCHER)
endif

if STARPU_MPI_CHECK
TESTS			=	$(starpu_mpi_TESTS)
endif

check_PROGRAMS = $(LOADER) $(starpu_mpi_TESTS)

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log

EXTRA_DIST = 				\
	helper.h			\
	user_defined_datatype_value.h

examplebindir = $(libdir)/starpu/examples/mpi

examplebin_PROGRAMS =

AM_CFLAGS += $(APP_CFLAGS)
AM_CXXFLAGS += $(APP_CXXFLAGS)
AM_FFLAGS += $(APP_FFLAGS)
AM_FCFLAGS += $(APP_FCFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include -I$(top_srcdir)/mpi/src -I$(top_srcdir)/src -I$(top_builddir)/src -I$(top_srcdir)/examples/ $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(STARPU_CUDA_LDFLAGS) $(STARPU_HIP_LDFLAGS)
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ ../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la $(STARPU_EXPORTED_LIBS)

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
	insert_task_can_execute			\
	insert_task_count			\
	insert_task_dyn_handles			\
	insert_task_node_choice			\
	insert_task_owner			\
	insert_task_owner2			\
	insert_task_owner_data			\
	insert_task_tags			\
	matrix					\
	matrix2					\
	mpi_barrier				\
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
	early_stuff				\
	star

if !STARPU_SIMGRID
# missing support in simgrid
starpu_mpi_TESTS +=				\
	attr					\
	display_bindings			\
	mpi_earlyrecv2				\
	mpi_earlyrecv2_sync			\
	block_interface				\
	block_interface_pinned			\
	ndim_interface				\
	insert_task_compute			\
	insert_task_sent_cache			\
	insert_task_recv_cache			\
	insert_task_seq				\
	tags_checking				\
	sync					\
	coop 					\
	coop_datatype 				\
	coop_large 				\
	coop_many				\
	coop_acknowledgement 			\
	coop_recv_not_yet_posted 		\
	coop_chained_sends 			\
	coop_wrong_order			\
	coop_without_task			\
	coop_user_defined_datatype		\
	coop_recv_wait_finalize			\
	coop_insert_task			\
	coop_cache				\
	mpi_task_submit

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
	mpi_barrier				\
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
	ndim_interface				\
	attr					\
	broadcast				\
	callback				\
	matrix					\
	matrix2					\
	insert_task				\
	insert_task_compute			\
	insert_task_sent_cache			\
	insert_task_recv_cache			\
	insert_task_can_execute			\
	insert_task_block			\
	insert_task_owner			\
	insert_task_owner2			\
	insert_task_owner_data			\
	insert_task_node_choice			\
	insert_task_count			\
	insert_task_dyn_handles			\
	insert_task_seq				\
	insert_task_tags			\
	multiple_send				\
	mpi_scatter_gather			\
	mpi_reduction				\
	user_defined_datatype			\
	tags_checking				\
	star					\
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
	driver 					\
	coop 					\
	coop_datatype 				\
	coop_large 				\
	coop_many 				\
	coop_acknowledgement 			\
	coop_recv_not_yet_posted 		\
	coop_chained_sends 			\
	coop_wrong_order			\
	coop_without_task			\
	coop_user_defined_datatype		\
	coop_recv_wait_finalize			\
	coop_insert_task			\
	coop_cache				\
	nothing					\
	display_bindings			\
	mpi_task_submit

if STARPU_USE_MPI_FT
noinst_PROGRAMS +=  \
	checkpoints
endif STARPU_USE_MPI_FT

XFAIL_TESTS=					\
	policy_register_toomany			\
	policy_unregister			\
	starpu_redefine				\
	nothing

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
if STARPU_USE_HIP
ring_SOURCES += ring_kernel_hip.hip
ring_sync_SOURCES += ring_kernel_hip.hip
ring_sync_detached_SOURCES += ring_kernel_hip.hip
ring_async_SOURCES += ring_kernel_hip.hip
ring_async_implicit_SOURCES += ring_kernel_hip.hip
insert_task_count_SOURCES += ring_kernel_hip.hip
endif
mpi_reduction_SOURCES = mpi_reduction.c
mpi_reduction_SOURCES += mpi_reduction_kernels.c
user_defined_datatype_SOURCES = user_defined_datatype.c
user_defined_datatype_SOURCES += ../../examples/interface/complex_interface.c

mpi_earlyrecv2_SOURCES = mpi_earlyrecv2.c
mpi_earlyrecv2_SOURCES += ../../examples/interface/complex_interface.c
mpi_earlyrecv2_sync_SOURCES = mpi_earlyrecv2_sync.c
mpi_earlyrecv2_sync_SOURCES += ../../examples/interface/complex_interface.c

coop_user_defined_datatype_SOURCES = coop_user_defined_datatype.c
coop_user_defined_datatype_SOURCES += ../../examples/interface/complex_interface.c

early_stuff_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)
