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

LIBS = $(top_builddir)/src/libstarpu.la @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/

EXTRA_DIST =					\
	microbenchs/null_kernel_gordon.c	\
	datawizard/sync_and_notify_data_gordon_kernels.c \
	datawizard/sync_and_notify_data_opencl_codelet.cl

CLEANFILES = 					\
	*.gcno *.gcda *.linkinfo		\
	microbenchs/null_kernel_gordon.spuelf	\
	datawizard/sync_and_notify_data_gordon_kernels.spuelf

BUILT_SOURCES =

if STARPU_USE_CUDA

# TODO define NVCCFLAGS
NVCC ?= nvcc

NVCCFLAGS += -I$(top_srcdir)/include/ -I$(top_builddir)/include

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

SUBDIRS = 

TESTS = $(check_PROGRAMS)

check_PROGRAMS =

check_PROGRAMS += 				\
	core/restart				\
	core/execute_on_a_specific_worker	\
	core/multithreaded			\
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
	core/tag_wait_api			\
	core/task_wait_api			\
	core/declare_deps_in_callback		\
	core/declare_deps_after_submission	\
	core/declare_deps_after_submission_synchronous	\
	core/get_current_task			\
	datawizard/data_implicit_deps		\
	datawizard/sync_and_notify_data		\
	datawizard/dsm_stress			\
	datawizard/write_only_tmp_buffer	\
	datawizard/dining_philosophers		\
	datawizard/readers_and_writers		\
	datawizard/unpartition			\
	datawizard/sync_with_data_with_mem	\
	datawizard/sync_with_data_with_mem_non_blocking\
	errorcheck/starpu_init_noworker		\
	errorcheck/invalid_blocking_calls	\
	errorcheck/invalid_tasks		\
	helper/cublas_init			\
	helper/pinned_memory			\
	helper/execute_on_all			\
	helper/starpu_create_sync_task		\
	microbenchs/async_tasks_overhead	\
	microbenchs/sync_tasks_overhead		\
	microbenchs/tasks_overhead		\
	microbenchs/prefetch_data_on_node 	\
	microbenchs/redundant_buffer		\
	microbenchs/local_pingpong		\
	overlap/overlap

core_restart_SOURCES =				\
	core/restart.c

core_execute_on_a_specific_worker_SOURCES = 	\
	core/execute_on_a_specific_worker.c

core_multithreaded_SOURCES =			\
	core/multithreaded.c

core_starpu_task_wait_for_all_SOURCES =		\
	core/starpu_task_wait_for_all.c

core_starpu_task_wait_SOURCES =			\
	core/starpu_task_wait.c

core_static_restartable_SOURCES =		\
	core/static_restartable.c

core_static_restartable_using_initializer_SOURCES =		\
	core/static_restartable_using_initializer.c

core_static_restartable_tag_SOURCES =		\
	core/static_restartable_tag.c

core_regenerate_SOURCES =			\
	core/regenerate.c

core_wait_all_regenerable_tasks_SOURCES =	\
	core/wait_all_regenerable_tasks.c

core_subgraph_repeat_SOURCES =			\
	core/subgraph_repeat.c

core_subgraph_repeat_regenerate_SOURCES =	\
	core/subgraph_repeat_regenerate.c

core_empty_task_SOURCES =			\
	core/empty_task.c

core_empty_task_sync_point_SOURCES =		\
	core/empty_task_sync_point.c

core_tag_wait_api_SOURCES =			\
	core/tag_wait_api.c

core_task_wait_api_SOURCES =			\
	core/task_wait_api.c

core_declare_deps_in_callback_SOURCES =		\
	core/declare_deps_in_callback.c

core_declare_deps_after_submission_SOURCES =	\
	core/declare_deps_after_submission.c

core_declare_deps_after_submission_synchronous_SOURCES =	\
	core/declare_deps_after_submission_synchronous.c

core_get_current_task_SOURCES =			\
	core/get_current_task.c


datawizard_data_implicit_deps_SOURCES =		\
	datawizard/data_implicit_deps.c

datawizard_dsm_stress_SOURCES =			\
	datawizard/dsm_stress.c

datawizard_write_only_tmp_buffer_SOURCES =	\
	datawizard/write_only_tmp_buffer.c

datawizard_dining_philosophers_SOURCES = 	\
	datawizard/dining_philosophers.c

datawizard_readers_and_writers_SOURCES = 	\
	datawizard/readers_and_writers.c

datawizard_unpartition_SOURCES =		\
	datawizard/unpartition.c

datawizard_sync_with_data_with_mem_SOURCES =	\
	datawizard/sync_with_data_with_mem.c

datawizard_sync_with_data_with_mem_non_blocking_SOURCES = \
	datawizard/sync_with_data_with_mem_non_blocking.c

errorcheck_starpu_init_noworker_SOURCES =	\
	errorcheck/starpu_init_noworker.c

errorcheck_invalid_blocking_calls_SOURCES =	\
	errorcheck/invalid_blocking_calls.c

errorcheck_invalid_tasks_SOURCES =		\
	errorcheck/invalid_tasks.c

helper_cublas_init_SOURCES =			\
	helper/cublas_init.c

helper_pinned_memory_SOURCES =			\
	helper/pinned_memory.c

helper_execute_on_all_SOURCES =			\
	helper/execute_on_all.c

helper_starpu_create_sync_task_SOURCES =	\
	helper/starpu_create_sync_task.c

microbenchs_async_tasks_overhead_SOURCES =	\
	microbenchs/async_tasks_overhead.c

microbenchs_sync_tasks_overhead_SOURCES =	\
	microbenchs/sync_tasks_overhead.c

microbenchs_tasks_overhead_SOURCES =		\
	microbenchs/tasks_overhead.c

microbenchs_prefetch_data_on_node_SOURCES =	\
	microbenchs/prefetch_data_on_node.c

datawizard_sync_and_notify_data_SOURCES =	\
	datawizard/sync_and_notify_data.c

nobase_STARPU_OPENCL_DATA_DATA = \
	datawizard/sync_and_notify_data_opencl_codelet.cl

microbenchs_redundant_buffer_SOURCES =		\
	microbenchs/redundant_buffer.c

microbenchs_local_pingpong_SOURCES =		\
	microbenchs/local_pingpong.c

if STARPU_USE_CUDA
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_kernels.cu
endif

if STARPU_USE_OPENCL
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_opencl.c
endif

if STARPU_USE_GORDON
datawizard_sync_and_notify_data_SOURCES +=	\
	datawizard/sync_and_notify_data_gordon_kernels.c

BUILT_SOURCES += 						\
	datawizard/sync_and_notify_data_gordon_kernels.spuelf	\
	microbenchs/null_kernel_gordon.spuelf
endif

overlap_overlap_SOURCES =			\
	overlap/overlap.c
