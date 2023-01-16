# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

DOX_DIR = $(top_builddir)/doc/doxygen_dev
DOX_CONFIG = $(top_srcdir)/doc/doxygen.cfg

DOX_MAIN_DIR = doxygen_dev
DOX_HTML_DIR = html_dev
DOX_LATEX_DIR = latex
DOX_PDF = starpu_dev.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	chapters/000_introduction.doxy \
	chapters/010_core.doxy

images =

if STARPU_BUILD_DOC
config.h: $(top_srcdir)/src/common/config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@
	@$(SED) -i '1s/^/\/\*\* \@file \*\/\n/' $@

dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	config.h					\
	chapters/version.sty				\
	chapters/version.html				\
	$(top_srcdir)/src/datawizard/data_request.h	\
	$(top_srcdir)/src/datawizard/coherency.h	\
	$(top_srcdir)/src/datawizard/sort_data_handles.h	\
	$(top_srcdir)/src/datawizard/memalloc.h	\
	$(top_srcdir)/src/datawizard/copy_driver.h	\
	$(top_srcdir)/src/datawizard/filters.h	\
	$(top_srcdir)/src/datawizard/datastats.h	\
	$(top_srcdir)/src/datawizard/write_back.h	\
	$(top_srcdir)/src/datawizard/interfaces/data_interface.h	\
	$(top_srcdir)/src/datawizard/memory_manager.h	\
	$(top_srcdir)/src/datawizard/node_ops.h	\
	$(top_srcdir)/src/datawizard/memstats.h	\
	$(top_srcdir)/src/datawizard/datawizard.h	\
	$(top_srcdir)/src/datawizard/memory_nodes.h	\
	$(top_srcdir)/src/datawizard/footprint.h	\
	$(top_srcdir)/src/datawizard/malloc.h	\
	$(top_srcdir)/src/drivers/cpu/driver_cpu.h	\
	$(top_srcdir)/src/drivers/cuda/driver_cuda.h	\
	$(top_srcdir)/src/drivers/opencl/driver_opencl_utils.h	\
	$(top_srcdir)/src/drivers/opencl/driver_opencl.h	\
	$(top_srcdir)/src/drivers/disk/driver_disk.h	\
	$(top_srcdir)/src/drivers/mpi/driver_mpi_common.h	\
	$(top_srcdir)/src/drivers/mpi/driver_mpi_sink.h	\
	$(top_srcdir)/src/drivers/mpi/driver_mpi_source.h	\
	$(top_srcdir)/src/drivers/mp_common/sink_common.h	\
	$(top_srcdir)/src/drivers/mp_common/mp_common.h	\
	$(top_srcdir)/src/drivers/mp_common/source_common.h	\
	$(top_srcdir)/src/drivers/driver_common/driver_common.h	\
	$(top_srcdir)/src/parallel_worker/starpu_parallel_worker_create.h	\
	$(top_srcdir)/src/profiling/profiling.h	\
	$(top_srcdir)/src/profiling/bound.h	\
	$(top_srcdir)/src/util/starpu_data_cpy.h	\
	$(top_srcdir)/src/util/openmp_runtime_support.h	\
	$(top_srcdir)/src/util/starpu_task_insert_utils.h	\
	$(top_srcdir)/src/common/graph.h	\
	$(top_srcdir)/src/common/fxt.h	\
	$(top_srcdir)/src/common/starpu_spinlock.h	\
	$(top_srcdir)/src/common/rbtree_i.h	\
	$(top_srcdir)/src/common/rbtree.h	\
	$(top_srcdir)/src/common/timing.h	\
	$(top_srcdir)/src/common/rwlock.h	\
	$(top_srcdir)/src/common/barrier.h	\
	$(top_srcdir)/src/common/prio_list.h	\
	$(top_srcdir)/src/common/barrier_counter.h	\
	$(top_srcdir)/src/common/uthash.h	\
	$(top_srcdir)/src/common/knobs.h	\
	$(top_srcdir)/src/common/utils.h	\
	$(top_srcdir)/src/common/thread.h	\
	$(top_srcdir)/src/common/list.h	\
	$(top_srcdir)/src/debug/starpu_debug_helpers.h	\
	$(top_srcdir)/src/debug/traces/starpu_fxt.h	\
	$(top_srcdir)/src/sched_policies/fifo_queues.h	\
	$(top_srcdir)/src/sched_policies/helper_mct.h	\
	$(top_srcdir)/src/sched_policies/sched_component.h	\
	$(top_srcdir)/src/sched_policies/prio_deque.h	\
	$(top_srcdir)/src/core/jobs.h	\
	$(top_srcdir)/src/core/disk_ops/unistd/disk_unistd_global.h	\
	$(top_srcdir)/src/core/dependencies/tags.h	\
	$(top_srcdir)/src/core/dependencies/data_concurrency.h	\
	$(top_srcdir)/src/core/dependencies/implicit_data_deps.h	\
	$(top_srcdir)/src/core/dependencies/cg.h	\
	$(top_srcdir)/src/core/idle_hook.h	\
	$(top_srcdir)/src/core/sched_ctx_list.h	\
	$(top_srcdir)/src/core/perfmodel/multiple_regression.h	\
	$(top_srcdir)/src/core/perfmodel/perfmodel.h	\
	$(top_srcdir)/src/core/perfmodel/regression.h	\
	$(top_srcdir)/src/core/debug.h	\
	$(top_srcdir)/src/core/sched_ctx.h	\
	$(top_srcdir)/src/core/simgrid.h	\
	$(top_srcdir)/src/core/task_bundle.h	\
	$(top_srcdir)/src/core/topology.h	\
	$(top_srcdir)/src/core/combined_workers.h	\
	$(top_srcdir)/src/core/detect_combined_workers.h	\
	$(top_srcdir)/src/core/task.h	\
	$(top_srcdir)/src/core/disk.h	\
	$(top_srcdir)/src/core/sched_policy.h	\
	$(top_srcdir)/src/core/errorcheck.h	\
	$(top_srcdir)/src/core/progress_hook.h	\
	$(top_srcdir)/src/core/drivers.h	\
	$(top_srcdir)/src/core/workers.h
endif

