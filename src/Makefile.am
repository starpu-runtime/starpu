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

SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo

lib_LTLIBRARIES = libstarpu.la

libstarpu_la_CPPFLAGS = -I$(top_srcdir)/include/

libstarpu_la_CFLAGS = -W -Wall -Wextra $(HWLOC_CFLAGS)
libstarpu_la_LIBADD = -lm $(HWLOC_LIBS)
libstarpu_la_LDFLAGS = --no-undefined

noinst_HEADERS = 						\
	core/dependencies/data_concurrency.h			\
	core/dependencies/cg.h					\
	core/dependencies/tags.h				\
	core/dependencies/htable.h				\
	core/policies/eager_central_priority_policy.h		\
	core/policies/sched_policy.h				\
	core/policies/random_policy.h				\
	core/policies/eager_central_policy.h			\
	core/policies/deque_modeling_policy.h			\
	core/policies/no_prio_policy.h				\
	core/policies/deque_modeling_policy_data_aware.h	\
	core/policies/work_stealing_policy.h			\
	core/mechanisms/priority_queues.h			\
	core/mechanisms/fifo_queues.h				\
	core/mechanisms/deque_queues.h				\
	core/mechanisms/queues.h				\
	core/mechanisms/stack_queues.h				\
	core/perfmodel/perfmodel.h				\
	core/perfmodel/regression.h				\
	core/jobs.h						\
	core/task.h						\
	core/workers.h						\
	core/topology.h						\
	core/debug.h						\
	core/errorcheck.h					\
	datawizard/footprint.h					\
	datawizard/datawizard.h					\
	datawizard/data_request.h				\
	datawizard/filters.h					\
	datawizard/write_back.h					\
	datawizard/datastats.h					\
	datawizard/memalloc.h					\
	datawizard/copy_driver.h				\
	datawizard/coherency.h					\
	datawizard/sort_data_handles.h				\
	datawizard/memory_nodes.h				\
	datawizard/interfaces/data_interface.h			\
	common/hash.h						\
	common/timing.h						\
	common/htable32.h					\
	common/list.h						\
	common/rwlock.h						\
	common/starpu_spinlock.h				\
	common/fxt.h						\
	common/utils.h						\
	drivers/cpu/driver_cpu.h				\
	drivers/gordon/driver_gordon.h				\
	drivers/gordon/gordon_interface.h			\
	drivers/cuda/driver_cuda.h

libstarpu_la_SOURCES = 						\
	common/hash.c 						\
	common/htable32.c					\
	common/rwlock.c						\
	common/starpu_spinlock.c				\
	common/timing.c						\
	common/fxt.c						\
	common/utils.c						\
	core/jobs.c						\
	core/task.c						\
	core/workers.c						\
	core/topology.c						\
	core/debug.c						\
	core/errorcheck.c					\
	core/progress_hook.c					\
	core/dependencies/cg.c					\
	core/dependencies/dependencies.c			\
	core/dependencies/tags.c				\
	core/dependencies/task_deps.c				\
	core/dependencies/htable.c				\
	core/dependencies/data_concurrency.c			\
	core/mechanisms/queues.c				\
	core/mechanisms/deque_queues.c				\
	core/mechanisms/priority_queues.c			\
	core/mechanisms/fifo_queues.c				\
	core/perfmodel/perfmodel_history.c			\
	core/perfmodel/perfmodel_bus.c				\
	core/perfmodel/perfmodel.c				\
	core/perfmodel/regression.c				\
	core/policies/no_prio_policy.c				\
	core/policies/eager_central_policy.c			\
	core/policies/eager_central_priority_policy.c		\
	core/policies/work_stealing_policy.c			\
	core/policies/sched_policy.c				\
	core/policies/deque_modeling_policy.c			\
	core/policies/random_policy.c				\
	core/policies/deque_modeling_policy_data_aware.c	\
	datawizard/memory_nodes.c				\
	datawizard/write_back.c					\
	datawizard/coherency.c					\
	datawizard/data_request.c				\
	datawizard/progress.c					\
	datawizard/copy_driver.c				\
	datawizard/filters.c					\
	datawizard/sort_data_handles.c				\
	datawizard/memalloc.c					\
	datawizard/footprint.c					\
	datawizard/datastats.c					\
	datawizard/user_interactions.c				\
	datawizard/interfaces/data_interface.c			\
	datawizard/interfaces/bcsr_interface.c			\
	datawizard/interfaces/csr_interface.c			\
	datawizard/interfaces/matrix_filters.c			\
	datawizard/interfaces/matrix_interface.c		\
	datawizard/interfaces/block_interface.c			\
	datawizard/interfaces/vector_interface.c		\
	datawizard/interfaces/bcsr_filters.c			\
	datawizard/interfaces/csr_filters.c			\
	datawizard/interfaces/vector_filters.c			\
	datawizard/interfaces/variable_interface.c		\
	util/malloc.c						\
	util/execute_on_all.c					\
	util/starpu_create_sync_task.c				\
	util/starpu_cublas.c					\
	util/file.c						\
	debug/latency.c

if STARPU_USE_CPU
libstarpu_la_SOURCES += drivers/cpu/driver_cpu.c
endif

if STARPU_USE_CUDA
libstarpu_la_SOURCES += drivers/cuda/driver_cuda.c
endif

if STARPU_USE_GORDON
libstarpu_la_SOURCES += drivers/gordon/driver_gordon.c
endif
