#
# StarPU
# Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

AM_CFLAGS = $(HWLOC_CFLAGS)
LIBS = $(top_builddir)/src/libstarpu.la $(HWLOC_LIBS) @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/src/

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
testbin_PROGRAMS =

SUBDIRS = 

TESTS = $(check_PROGRAMS)

if STARPU_COVERAGE_ENABLED
TESTS	+=	coverage/coverage.sh
endif

check_PROGRAMS =

check_PROGRAMS += 				\
	core/restart				\
	core/execute_on_a_specific_worker	\
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
	datawizard/acquire_release		\
	datawizard/acquire_release2		\
	datawizard/data_implicit_deps		\
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

testbin_PROGRAMS +=				\
	core/restart
core_restart_SOURCES =				\
	core/restart.c

testbin_PROGRAMS +=				\
	core/execute_on_a_specific_worker
core_execute_on_a_specific_worker_SOURCES = 	\
	core/execute_on_a_specific_worker.c

testbin_PROGRAMS +=				\
	core/multithreaded
core_multithreaded_SOURCES =			\
	core/multithreaded.c

testbin_PROGRAMS +=				\
	core/multithreaded_init
core_multithreaded_init_SOURCES =			\
	core/multithreaded_init.c

testbin_PROGRAMS +=				\
	core/starpu_task_wait_for_all
core_starpu_task_wait_for_all_SOURCES =		\
	core/starpu_task_wait_for_all.c

testbin_PROGRAMS +=				\
	core/starpu_task_wait
core_starpu_task_wait_SOURCES =			\
	core/starpu_task_wait.c

testbin_PROGRAMS +=				\
	core/static_restartable
core_static_restartable_SOURCES =		\
	core/static_restartable.c

testbin_PROGRAMS +=				\
	core/static_restartable_using_initializer
core_static_restartable_using_initializer_SOURCES =		\
	core/static_restartable_using_initializer.c

testbin_PROGRAMS +=				\
	core/static_restartable_tag
core_static_restartable_tag_SOURCES =		\
	core/static_restartable_tag.c

testbin_PROGRAMS +=				\
	core/regenerate
core_regenerate_SOURCES =			\
	core/regenerate.c

testbin_PROGRAMS +=				\
	core/wait_all_regenerable_tasks
core_wait_all_regenerable_tasks_SOURCES =	\
	core/wait_all_regenerable_tasks.c

testbin_PROGRAMS +=				\
	core/subgraph_repeat
core_subgraph_repeat_SOURCES =			\
	core/subgraph_repeat.c

testbin_PROGRAMS +=				\
	core/subgraph_repeat_regenerate
core_subgraph_repeat_regenerate_SOURCES =	\
	core/subgraph_repeat_regenerate.c

testbin_PROGRAMS +=				\
	core/empty_task
core_empty_task_SOURCES =			\
	core/empty_task.c

testbin_PROGRAMS +=				\
	core/empty_task_sync_point
core_empty_task_sync_point_SOURCES =		\
	core/empty_task_sync_point.c

testbin_PROGRAMS +=				\
	core/empty_task_sync_point_tasks
core_empty_task_sync_point_tasks_SOURCES =	\
	core/empty_task_sync_point_tasks.c

testbin_PROGRAMS +=				\
	core/empty_task_chain
core_empty_task_chain_SOURCES =			\
	core/empty_task_chain.c

testbin_PROGRAMS +=				\
	core/tag_wait_api
core_tag_wait_api_SOURCES =			\
	core/tag_wait_api.c

testbin_PROGRAMS +=				\
	core/task_wait_api
core_task_wait_api_SOURCES =			\
	core/task_wait_api.c

testbin_PROGRAMS +=				\
	core/declare_deps_in_callback
core_declare_deps_in_callback_SOURCES =		\
	core/declare_deps_in_callback.c

testbin_PROGRAMS +=				\
	core/declare_deps_after_submission
core_declare_deps_after_submission_SOURCES =	\
	core/declare_deps_after_submission.c

testbin_PROGRAMS +=				\
	core/declare_deps_after_submission_synchronous
core_declare_deps_after_submission_synchronous_SOURCES =	\
	core/declare_deps_after_submission_synchronous.c

testbin_PROGRAMS +=				\
	core/get_current_task
core_get_current_task_SOURCES =			\
	core/get_current_task.c

testbin_PROGRAMS +=				\
	datawizard/acquire_release
datawizard_acquire_release_SOURCES =		\
	datawizard/acquire_release.c
if STARPU_USE_CUDA
datawizard_acquire_release_SOURCES +=		\
	datawizard/acquire_release_cuda.cu
endif

testbin_PROGRAMS +=				\
	datawizard/acquire_release2
datawizard_acquire_release2_SOURCES =		\
	datawizard/acquire_release2.c
if STARPU_USE_CUDA
datawizard_acquire_release2_SOURCES +=		\
	datawizard/acquire_release_cuda.cu
endif

testbin_PROGRAMS +=				\
	datawizard/data_implicit_deps
datawizard_data_implicit_deps_SOURCES =		\
	datawizard/data_implicit_deps.c

testbin_PROGRAMS +=				\
	datawizard/scratch
datawizard_scratch_SOURCES =			\
	datawizard/scratch.c

if STARPU_USE_CUDA
datawizard_scratch_SOURCES +=		\
	datawizard/scratch_cuda.cu
endif

testbin_PROGRAMS +=				\
	datawizard/dsm_stress
datawizard_dsm_stress_SOURCES =			\
	datawizard/dsm_stress.c

testbin_PROGRAMS +=				\
	datawizard/write_only_tmp_buffer
datawizard_write_only_tmp_buffer_SOURCES =	\
	datawizard/write_only_tmp_buffer.c

testbin_PROGRAMS +=				\
	datawizard/data_invalidation
datawizard_data_invalidation_SOURCES =	\
	datawizard/data_invalidation.c

testbin_PROGRAMS +=				\
	datawizard/dining_philosophers
datawizard_dining_philosophers_SOURCES = 	\
	datawizard/dining_philosophers.c

testbin_PROGRAMS +=				\
	datawizard/manual_reduction
datawizard_manual_reduction_SOURCES = 		\
	datawizard/manual_reduction.c

testbin_PROGRAMS +=				\
	datawizard/readers_and_writers
datawizard_readers_and_writers_SOURCES = 	\
	datawizard/readers_and_writers.c

testbin_PROGRAMS +=				\
	datawizard/unpartition
datawizard_unpartition_SOURCES =		\
	datawizard/unpartition.c

testbin_PROGRAMS +=				\
	datawizard/user_interaction_implicit
datawizard_user_interaction_implicit_SOURCES =	\
	datawizard/user_interaction_implicit.c

testbin_PROGRAMS +=				\
	datawizard/reclaim
datawizard_reclaim_SOURCES =			\
	datawizard/reclaim.c

testbin_PROGRAMS +=				\
	datawizard/sync_with_data_with_mem
datawizard_sync_with_data_with_mem_SOURCES =	\
	datawizard/sync_with_data_with_mem.c

testbin_PROGRAMS +=				\
	datawizard/sync_with_data_with_mem_non_blocking
datawizard_sync_with_data_with_mem_non_blocking_SOURCES = \
	datawizard/sync_with_data_with_mem_non_blocking.c

testbin_PROGRAMS +=				\
	datawizard/sync_with_data_with_mem_non_blocking_implicit
datawizard_sync_with_data_with_mem_non_blocking_implicit_SOURCES = \
	datawizard/sync_with_data_with_mem_non_blocking_implicit.c

testbin_PROGRAMS +=				\
	datawizard/mpi_like
datawizard_mpi_like_SOURCES =			\
	datawizard/mpi_like.c

testbin_PROGRAMS +=				\
	datawizard/mpi_like_async
datawizard_mpi_like_async_SOURCES =		\
	datawizard/mpi_like_async.c

testbin_PROGRAMS +=				\
	datawizard/critical_section_with_void_interface
datawizard_critical_section_with_void_interface_SOURCES =	\
	datawizard/critical_section_with_void_interface.c

testbin_PROGRAMS +=				\
	datawizard/increment_redux

datawizard_increment_redux_SOURCES =		\
	datawizard/increment_redux.c

testbin_PROGRAMS +=				\
	datawizard/increment_redux_v2

datawizard_increment_redux_v2_SOURCES =		\
	datawizard/increment_redux_v2.c

if STARPU_USE_CUDA
datawizard_mpi_like_SOURCES +=			\
	datawizard/cuda_codelet_unsigned_inc.cu

datawizard_mpi_like_async_SOURCES +=		\
	datawizard/cuda_codelet_unsigned_inc.cu
endif

testbin_PROGRAMS +=				\
	errorcheck/starpu_init_noworker
errorcheck_starpu_init_noworker_SOURCES =	\
	errorcheck/starpu_init_noworker.c

testbin_PROGRAMS +=				\
	errorcheck/invalid_blocking_calls
errorcheck_invalid_blocking_calls_SOURCES =	\
	errorcheck/invalid_blocking_calls.c

testbin_PROGRAMS +=				\
	errorcheck/invalid_tasks
errorcheck_invalid_tasks_SOURCES =		\
	errorcheck/invalid_tasks.c

testbin_PROGRAMS +=				\
	helper/cublas_init
helper_cublas_init_SOURCES =			\
	helper/cublas_init.c

testbin_PROGRAMS +=				\
	helper/starpu_data_cpy
helper_starpu_data_cpy_SOURCES =		\
	helper/starpu_data_cpy.c	

testbin_PROGRAMS +=				\
	helper/pinned_memory
helper_pinned_memory_SOURCES =			\
	helper/pinned_memory.c

testbin_PROGRAMS +=				\
	helper/execute_on_all
helper_execute_on_all_SOURCES =			\
	helper/execute_on_all.c

testbin_PROGRAMS +=				\
	helper/starpu_create_sync_task
helper_starpu_create_sync_task_SOURCES =	\
	helper/starpu_create_sync_task.c

testbin_PROGRAMS +=				\
	microbenchs/async_tasks_overhead
microbenchs_async_tasks_overhead_SOURCES =	\
	microbenchs/async_tasks_overhead.c

testbin_PROGRAMS +=				\
	microbenchs/sync_tasks_overhead
microbenchs_sync_tasks_overhead_SOURCES =	\
	microbenchs/sync_tasks_overhead.c

testbin_PROGRAMS +=				\
	microbenchs/tasks_overhead
microbenchs_tasks_overhead_SOURCES =		\
	microbenchs/tasks_overhead.c

testbin_PROGRAMS +=				\
	microbenchs/prefetch_data_on_node
microbenchs_prefetch_data_on_node_SOURCES =	\
	microbenchs/prefetch_data_on_node.c

testbin_PROGRAMS +=				\
	datawizard/sync_and_notify_data
datawizard_sync_and_notify_data_SOURCES =	\
	datawizard/sync_and_notify_data.c

testbin_PROGRAMS +=				\
	datawizard/sync_and_notify_data_implicit
datawizard_sync_and_notify_data_implicit_SOURCES =	\
	datawizard/sync_and_notify_data_implicit.c

testbin_PROGRAMS +=				\
	microbenchs/redundant_buffer
microbenchs_redundant_buffer_SOURCES =		\
	microbenchs/redundant_buffer.c

testbin_PROGRAMS +=				\
	microbenchs/display_structures_size
microbenchs_display_structures_size_SOURCES =	\
	microbenchs/display_structures_size.c

testbin_PROGRAMS +=				\
	microbenchs/local_pingpong
microbenchs_local_pingpong_SOURCES =		\
	microbenchs/local_pingpong.c

if STARPU_USE_CUDA
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_kernels.cu

datawizard_sync_and_notify_data_implicit_SOURCES +=	\
	datawizard/sync_and_notify_data_kernels.cu
endif

if STARPU_USE_OPENCL
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_opencl.c

datawizard_sync_and_notify_data_implicit_SOURCES +=	\
	datawizard/sync_and_notify_data_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/sync_and_notify_data_opencl_codelet.cl
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

testbin_PROGRAMS +=				\
	overlap/overlap
overlap_overlap_SOURCES =			\
	overlap/overlap.c

testbin_PROGRAMS +=					\
	parallel_tasks/explicit_combined_worker
parallel_tasks_explicit_combined_worker_SOURCES =	\
	parallel_tasks/explicit_combined_worker.c

testbin_PROGRAMS +=				\
	parallel_tasks/parallel_kernels
parallel_tasks_parallel_kernels_SOURCES =	\
	parallel_tasks/parallel_kernels.c

testbin_PROGRAMS +=				\
	parallel_tasks/parallel_kernels_spmd
parallel_tasks_parallel_kernels_spmd_SOURCES =	\
	parallel_tasks/parallel_kernels_spmd.c

testbin_PROGRAMS +=				\
	perfmodels/regression_based
perfmodels_regression_based_SOURCES = 		\
	perfmodels/regression_based.c

testbin_PROGRAMS +=				\
	perfmodels/non_linear_regression_based
perfmodels_non_linear_regression_based_SOURCES = \
	perfmodels/non_linear_regression_based.c
