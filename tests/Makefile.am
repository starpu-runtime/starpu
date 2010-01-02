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
	microbenchs/sync_and_notify_data_gordon_kernels.c

CLEANFILES = 					\
	*.gcno *.gcda *.linkinfo		\
	microbenchs/null_kernel_gordon.spuelf	\
	microbenchs/sync_and_notify_data_gordon_kernels.spuelf

BUILT_SOURCES =

if USE_CUDA

# TODO define NVCCFLAGS
NVCC ?= nvcc

NVCCFLAGS += -I$(top_srcdir)/include/

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS)

.cu.o:
	$(MKDIR_P) `dirname $@`
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir}

endif

if USE_GORDON

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
	core/starpu_wait_all_tasks		\
	core/starpu_wait_task			\
	core/static_restartable			\
	core/static_restartable_using_initializer\
	core/static_restartable_tag		\
	core/empty_task				\
	core/empty_task_sync_point		\
	datawizard/write_only_tmp_buffer	\
	datawizard/dining_philosophers		\
	datawizard/readers_and_writers		\
	datawizard/unpartition			\
	errorcheck/starpu_init_noworker		\
	errorcheck/invalid_blocking_calls	\
	helper/cublas_init			\
	helper/pinned_memory			\
	helper/execute_on_all			\
	microbenchs/async-tasks-overhead	\
	microbenchs/sync-tasks-overhead		\
	microbenchs/tasks-overhead		\
	microbenchs/tag-wait-api		\
	microbenchs/dsm_stress			\
	microbenchs/prefetch_data_on_node 	\
	microbenchs/sync_and_notify_data	\
	microbenchs/redundant_buffer		\
	overlap/overlap

core_restart_SOURCES =				\
	core/restart.c

core_execute_on_a_specific_worker_SOURCES = 	\
	core/execute_on_a_specific_worker.c

core_multithreaded_SOURCES =			\
	core/multithreaded.c

core_starpu_wait_all_tasks_SOURCES =		\
	core/starpu_wait_all_tasks.c

core_starpu_wait_task_SOURCES =			\
	core/starpu_wait_task.c

core_static_restartable_SOURCES =		\
	core/static_restartable.c

core_static_restartable_using_initializer_SOURCES =		\
	core/static_restartable_using_initializer.c

core_static_restartable_tag_SOURCES =		\
	core/static_restartable_tag.c

core_empty_task_SOURCES =			\
	core/empty_task.c

core_empty_task_sync_point_SOURCES =		\
	core/empty_task_sync_point.c

datawizard_write_only_tmp_buffer_SOURCES =	\
	datawizard/write_only_tmp_buffer.c

datawizard_dining_philosophers_SOURCES = 	\
	datawizard/dining_philosophers.c

datawizard_readers_and_writers_SOURCES = 	\
	datawizard/readers_and_writers.c

datawizard_unpartition_SOURCES =		\
	datawizard/unpartition.c

errorcheck_starpu_init_noworker_SOURCES =	\
	errorcheck/starpu_init_noworker.c

errorcheck_invalid_blocking_calls_SOURCES =	\
	errorcheck/invalid_blocking_calls.c

helper_cublas_init_SOURCES =			\
	helper/cublas_init.c

helper_pinned_memory_SOURCES =			\
	helper/pinned_memory.c

helper_execute_on_all_SOURCES =			\
	helper/execute_on_all.c

microbenchs_async_tasks_overhead_SOURCES =	\
	microbenchs/async-tasks-overhead.c

microbenchs_sync_tasks_overhead_SOURCES =	\
	microbenchs/sync-tasks-overhead.c

microbenchs_tasks_overhead_SOURCES =		\
	microbenchs/tasks-overhead.c

microbenchs_tag_wait_api_SOURCES =		\
	microbenchs/tag-wait-api.c

microbenchs_dsm_stress_SOURCES =		\
	microbenchs/dsm_stress.c

microbenchs_prefetch_data_on_node_SOURCES =	\
	microbenchs/prefetch_data_on_node.c

microbenchs_sync_and_notify_data_SOURCES =	\
	microbenchs/sync_and_notify_data.c

microbenchs_redundant_buffer_SOURCES =		\
	microbenchs/redundant_buffer.c

if USE_CUDA
microbenchs_sync_and_notify_data_SOURCES +=	\
	microbenchs/sync_and_notify_data_kernels.cu
endif

if USE_GORDON
microbenchs_sync_and_notify_data_SOURCES +=	\
	microbenchs/sync_and_notify_data_gordon_kernels.c

BUILT_SOURCES += 						\
	microbenchs/sync_and_notify_data_gordon_kernels.spuelf	\
	microbenchs/null_kernel_gordon.spuelf
endif

overlap_overlap_SOURCES =			\
	overlap/overlap.c
