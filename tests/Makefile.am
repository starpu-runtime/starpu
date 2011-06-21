# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
# Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
# Copyright (C) 2010, 2011  Institut National de Recherche en Informatique et Automatique
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

AM_CFLAGS = $(HWLOC_CFLAGS) -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS)
LIBS = $(top_builddir)/src/libstarpu.la $(HWLOC_LIBS) @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/src -I$(top_srcdir)/src/
AM_LDFLAGS = $(STARPU_CUDA_LDFLAGS) $(STARPU_OPENCL_LDFLAGS)

EXTRA_DIST =					\
	microbenchs/null_kernel_gordon.c	\
	datawizard/sync_and_notify_data_gordon_kernels.c \
	datawizard/sync_and_notify_data_opencl_codelet.cl\
	coverage/coverage.sh

CLEANFILES = 					\
	*.gcno *.gcda *.linkinfo		\
	microbenchs/null_kernel_gordon.spuelf	\
	datawizard/sync_and_notify_data_gordon_kernels.spuelf

BUILT_SOURCES =
SUBDIRS =

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA =
endif

if STARPU_USE_CUDA

# TODO define NVCCFLAGS
NVCC ?= nvcc

NVCCFLAGS += -I$(top_srcdir)/include/ -I$(top_builddir)/include $(HWLOC_CFLAGS)

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS)

.cu.o:
	$(MKDIR_P) `dirname $@`
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir}

endif

if STARPU_USE_GORDON

SPU_CC ?= spu-gcc
SPU_LD ?= spu-ld

.c.spuo:
	$(MKDIR_P) `dirname $@`
	$(SPU_CC) -c -fpic $< -o $@

.spuo.spuelf:
	$(MKDIR_P) `dirname $@`
	$(SPU_LD) $< -o $@

#BUILT_SOURCES +=
#	microbenchs/null_kernel_gordon.spuelf

endif

testbindir = $(libdir)/starpu/tests

#####################################
# What to install and what to check #
#####################################

if !STARPU_HAVE_WINDOWS
## test loader program
LOADER			=	loader
LOADER_BIN		=	$(abs_top_builddir)/tests/$(LOADER)
TESTS_ENVIRONMENT	=	$(LOADER_BIN)
endif

TESTS = $(testbin_PROGRAMS)

if STARPU_COVERAGE_ENABLED
TESTS	+=	coverage/coverage.sh
endif

testbin_PROGRAMS =				\
	core/restart				\
	core/execute_on_a_specific_worker	\
	core/insert_task			\
	core/multithreaded			\
	core/multithreaded_init			\
	core/starpu_task_wait_for_all		\
	core/starpu_task_wait			\
	core/static_restartable			\
	core/static_restartable_using_initializer\
	core/static_restartable_tag		\
	core/regenerate				\
	core/wait_all_regenerable_tasks		\
	core/subgraph_repeat			\
	core/subgraph_repeat_regenerate		\
	core/empty_task				\
	core/empty_task_sync_point		\
	core/empty_task_sync_point_tasks	\
	core/empty_task_chain			\
	core/tag_wait_api			\
	core/task_wait_api			\
	core/declare_deps_in_callback		\
	core/declare_deps_after_submission	\
	core/declare_deps_after_submission_synchronous	\
	core/get_current_task			\
	datawizard/acquire_cb			\
	datawizard/acquire_cb_insert		\
	datawizard/acquire_release		\
	datawizard/acquire_release2		\
	datawizard/data_implicit_deps		\
	datawizard/data_lookup			\
	datawizard/scratch			\
	datawizard/sync_and_notify_data		\
	datawizard/sync_and_notify_data_implicit\
	datawizard/dsm_stress			\
	datawizard/write_only_tmp_buffer	\
	datawizard/data_invalidation		\
	datawizard/dining_philosophers		\
	datawizard/manual_reduction		\
	datawizard/readers_and_writers		\
	datawizard/unpartition			\
	datawizard/user_interaction_implicit	\
	datawizard/reclaim			\
	datawizard/sync_with_data_with_mem	\
	datawizard/sync_with_data_with_mem_non_blocking\
	datawizard/sync_with_data_with_mem_non_blocking_implicit\
	datawizard/mpi_like			\
	datawizard/mpi_like_async		\
	datawizard/critical_section_with_void_interface\
	datawizard/increment_redux		\
	datawizard/increment_redux_v2		\
	datawizard/handle_to_pointer		\
	datawizard/lazy_allocation		\
	errorcheck/starpu_init_noworker		\
	errorcheck/invalid_blocking_calls	\
	errorcheck/invalid_tasks		\
	helper/cublas_init			\
	helper/starpu_data_cpy			\
	helper/pinned_memory			\
	helper/execute_on_all			\
	helper/starpu_create_sync_task		\
	microbenchs/async_tasks_overhead	\
	microbenchs/sync_tasks_overhead		\
	microbenchs/tasks_overhead		\
	microbenchs/prefetch_data_on_node 	\
	microbenchs/redundant_buffer		\
	microbenchs/local_pingpong		\
	overlap/overlap				\
	parallel_tasks/explicit_combined_worker	\
	parallel_tasks/parallel_kernels		\
	parallel_tasks/parallel_kernels_spmd	\
	perfmodels/regression_based		\
	perfmodels/non_linear_regression_based

if STARPU_HAVE_WINDOWS
check_PROGRAMS = $(testbin_PROGRAMS)
else
check_PROGRAMS = $(LOADER) $(testbin_PROGRAMS)
endif

#######################
# Source files        #
#######################

datawizard_acquire_release_SOURCES =		\
	datawizard/acquire_release.c
if STARPU_USE_CUDA
datawizard_acquire_release_SOURCES +=		\
	datawizard/acquire_release_cuda.cu
endif

datawizard_acquire_release2_SOURCES =		\
	datawizard/acquire_release2.c
if STARPU_USE_CUDA
datawizard_acquire_release2_SOURCES +=		\
	datawizard/acquire_release_cuda.cu
endif

datawizard_scratch_SOURCES =			\
	datawizard/scratch.c
if STARPU_USE_CUDA
datawizard_scratch_SOURCES +=		\
	datawizard/scratch_cuda.cu
endif

datawizard_mpi_like_SOURCES =		\
	datawizard/mpi_like.c
if STARPU_USE_CUDA
datawizard_mpi_like_SOURCES +=			\
	datawizard/cuda_codelet_unsigned_inc.cu
endif

datawizard_mpi_like_async_SOURCES =		\
	datawizard/mpi_like_async.c
if STARPU_USE_CUDA
datawizard_mpi_like_async_SOURCES +=		\
	datawizard/cuda_codelet_unsigned_inc.cu
endif

datawizard_sync_and_notify_data_SOURCES =	\
	datawizard/sync_and_notify_data.c
if STARPU_USE_CUDA
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_kernels.cu
endif
if STARPU_USE_OPENCL
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/sync_and_notify_data_opencl_codelet.cl
endif

datawizard_sync_and_notify_data_implicit_SOURCES =	\
	datawizard/sync_and_notify_data_implicit.c
if STARPU_USE_CUDA
datawizard_sync_and_notify_data_implicit_SOURCES +=	\
	datawizard/sync_and_notify_data_kernels.cu
endif
if STARPU_USE_OPENCL
datawizard_sync_and_notify_data_implicit_SOURCES +=	\
	datawizard/sync_and_notify_data_opencl.c
endif

if STARPU_USE_GORDON
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_gordon_kernels.c
datawizard_sync_and_notify_data_implicit_SOURCES +=	\
	datawizard/sync_and_notify_data_gordon_kernels.c
BUILT_SOURCES += 						\
	datawizard/sync_and_notify_data_gordon_kernels.spuelf	\
	microbenchs/null_kernel_gordon.spuelf
endif
