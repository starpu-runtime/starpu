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

noinst_HEADERS = 						\
	core/dependencies/data-concurrency.h			\
	core/dependencies/tags.h				\
	core/dependencies/htable.h				\
	core/policies/eager-central-priority-policy.h		\
	core/policies/sched_policy.h				\
	core/policies/random-policy.h				\
	core/policies/eager-central-policy.h			\
	core/policies/deque-modeling-policy.h			\
	core/policies/no-prio-policy.h				\
	core/policies/deque-modeling-policy-data-aware.h	\
	core/policies/work-stealing-policy.h			\
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
	datawizard/hierarchy.h					\
	datawizard/progress.h					\
	datawizard/write_back.h					\
	datawizard/datastats.h					\
	datawizard/memalloc.h					\
	datawizard/data_parameters.h				\
	datawizard/copy-driver.h				\
	datawizard/coherency.h					\
	datawizard/memory_nodes.h				\
	datawizard/interfaces/data_interface.h			\
	common/hash.h						\
	common/timing.h						\
	common/htable32.h					\
	common/list.h						\
	common/rwlock.h						\
	common/starpu-spinlock.h				\
	common/fxt.h						\
	drivers/core/driver_core.h				\
	drivers/gordon/driver_gordon.h				\
	drivers/gordon/gordon_interface.h			\
	drivers/cuda/driver_cuda.h

libstarpu_la_SOURCES = 						\
	common/hash.c 						\
	common/htable32.c					\
	common/rwlock.c						\
	common/starpu-spinlock.c				\
	common/timing.c						\
	common/fxt.c						\
	core/jobs.c						\
	core/task.c						\
	core/workers.c						\
	core/topology.c						\
	core/debug.c						\
	core/errorcheck.c					\
	core/dependencies/tags.c				\
	core/dependencies/htable.c				\
	core/dependencies/data-concurrency.c			\
	core/mechanisms/queues.c				\
	core/mechanisms/deque_queues.c				\
	core/mechanisms/priority_queues.c			\
	core/mechanisms/fifo_queues.c				\
	core/perfmodel/perfmodel_history.c			\
	core/perfmodel/perfmodel_bus.c				\
	core/perfmodel/perfmodel.c				\
	core/perfmodel/regression.c				\
	core/policies/no-prio-policy.c				\
	core/policies/eager-central-policy.c			\
	core/policies/eager-central-priority-policy.c		\
	core/policies/work-stealing-policy.c			\
	core/policies/sched_policy.c				\
	core/policies/deque-modeling-policy.c			\
	core/policies/random-policy.c				\
	core/policies/deque-modeling-policy-data-aware.c	\
	datawizard/memory_nodes.c				\
	datawizard/write_back.c					\
	datawizard/coherency.c					\
	datawizard/data_request.c				\
	datawizard/progress.c					\
	datawizard/copy-driver.c				\
	datawizard/hierarchy.c					\
	datawizard/memalloc.c					\
	datawizard/footprint.c					\
	datawizard/datastats.c					\
	datawizard/user_interactions.c				\
	datawizard/interfaces/bcsr_interface.c			\
	datawizard/interfaces/csr_interface.c			\
	datawizard/interfaces/blas_filters.c			\
	datawizard/interfaces/blas_interface.c			\
	datawizard/interfaces/block_interface.c			\
	datawizard/interfaces/vector_interface.c		\
	datawizard/interfaces/bcsr_filters.c			\
	datawizard/interfaces/csr_filters.c			\
	datawizard/interfaces/vector_filters.c			\
	util/malloc.c						\
	util/execute_on_all.c					\
	util/starpu_cublas.c
	
if USE_CPU
libstarpu_la_SOURCES += drivers/core/driver_core.c
endif

if USE_CUDA
libstarpu_la_SOURCES += drivers/cuda/driver_cuda.c
endif

if USE_GORDON
libstarpu_la_SOURCES += drivers/gordon/driver_gordon.c
endif
