# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2013       Simon Archipoff
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

include $(top_srcdir)/starpu-notests.mk

SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo

EXTRA_DIST = dolib.c

xml_DATA = $(srcdir)/core/perfmodel/starpu-perfmodel.dtd
xmldir = $(pkgdatadir)
EXTRA_DIST += core/perfmodel/starpu-perfmodel.dtd

ldflags =

libstarpu_so_version = $(LIBSTARPU_INTERFACE_CURRENT):$(LIBSTARPU_INTERFACE_REVISION):$(LIBSTARPU_INTERFACE_AGE)

if STARPU_HAVE_WINDOWS

LC_MESSAGES=C
export LC_MESSAGES

ldflags += -Xlinker --output-def -Xlinker .libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.def

if STARPU_HAVE_MS_LIB
.libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.lib: libstarpu-@STARPU_EFFECTIVE_VERSION@.la dolib
	./dolib "$(STARPU_MS_LIB)" $(STARPU_MS_LIB_ARCH) .libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.def @STARPU_EFFECTIVE_VERSION@ $(libstarpu_so_version) .libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.lib
all-local: .libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.lib
endif STARPU_HAVE_MS_LIB

install-exec-hook:
	$(INSTALL) .libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.def $(DESTDIR)$(libdir)
if STARPU_HAVE_MS_LIB
	$(INSTALL) .libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.lib $(DESTDIR)$(libdir)
	$(INSTALL) .libs/libstarpu-@STARPU_EFFECTIVE_VERSION@.exp $(DESTDIR)$(libdir)
endif STARPU_HAVE_MS_LIB

endif STARPU_HAVE_WINDOWS

lib_LTLIBRARIES = libstarpu-@STARPU_EFFECTIVE_VERSION@.la

libstarpu_@STARPU_EFFECTIVE_VERSION@_la_CPPFLAGS = -I$(top_srcdir)/include/ -DBUILDING_STARPU  $(GLOBAL_AM_CFLAGS) $(HWLOC_CFLAGS) $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(STARPU_COI_CPPFLAGS) $(STARPU_SCIF_CPPFLAGS) $(STARPU_RCCE_CFLAGS) $(STARPU_RCCE_CPPFLAGS) -DSTARPU_DATADIR='"$(datadir)"'

libstarpu_@STARPU_EFFECTIVE_VERSION@_la_LIBADD = -lm $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(LIBSTARPU_LDFLAGS)
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS = $(ldflags) $(FXT_LDFLAGS) -no-undefined									\
  -version-info $(libstarpu_so_version)

if STARPU_HAVE_DARWIN
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS += \
							-Wl,-U,_starpu_main \
							-Wl,-U,_smpi_main \
							-Wl,-U,__starpu_mpi_simgrid_init \
							-Wl,-U,_smpi_simulated_main_ \
							-Wl,-U,_starpu_mpi_world_rank
endif

noinst_HEADERS = 						\
	core/dependencies/data_concurrency.h			\
	core/dependencies/cg.h					\
	core/dependencies/tags.h				\
	core/dependencies/implicit_data_deps.h			\
	core/disk.h						\
	core/disk_ops/unistd/disk_unistd_global.h		\
	core/progress_hook.h                                    \
	core/idle_hook.h                                        \
	core/sched_policy.h					\
	core/sched_ctx.h					\
	core/sched_ctx_list.h					\
	core/perfmodel/perfmodel.h				\
	core/perfmodel/regression.h				\
	core/perfmodel/multiple_regression.h			\
	core/jobs.h						\
	core/task.h						\
	core/drivers.h						\
	core/workers.h						\
	core/topology.h						\
	core/debug.h						\
	core/errorcheck.h					\
	core/combined_workers.h					\
	core/simgrid.h						\
	core/task_bundle.h					\
	core/detect_combined_workers.h				\
	sched_policies/helper_mct.h				\
	sched_policies/fifo_queues.h				\
	datawizard/node_ops.h					\
	datawizard/footprint.h					\
	datawizard/datawizard.h					\
	datawizard/data_request.h				\
	datawizard/filters.h					\
	datawizard/write_back.h					\
	datawizard/datastats.h					\
	datawizard/malloc.h					\
	datawizard/memstats.h					\
	datawizard/memory_manager.h				\
	datawizard/memalloc.h					\
	datawizard/copy_driver.h				\
	datawizard/coherency.h					\
	datawizard/sort_data_handles.h				\
	datawizard/memory_nodes.h				\
	datawizard/interfaces/data_interface.h			\
	common/barrier.h					\
	common/timing.h						\
	common/list.h						\
	common/rwlock.h						\
	common/starpu_spinlock.h				\
	common/fxt.h						\
	common/utils.h						\
	common/thread.h						\
	common/barrier.h					\
	common/uthash.h						\
	common/barrier_counter.h				\
	common/rbtree.h						\
	common/rbtree_i.h					\
	common/prio_list.h					\
	common/graph.h						\
	drivers/driver_common/driver_common.h			\
	drivers/mp_common/mp_common.h				\
	drivers/mp_common/source_common.h			\
	drivers/mp_common/sink_common.h				\
	drivers/cpu/driver_cpu.h				\
	drivers/cuda/driver_cuda.h				\
	drivers/opencl/driver_opencl.h				\
	drivers/opencl/driver_opencl_utils.h			\
	debug/starpu_debug_helpers.h				\
	drivers/mic/driver_mic_common.h				\
	drivers/mic/driver_mic_source.h				\
	drivers/mic/driver_mic_sink.h				\
	drivers/mpi/driver_mpi_common.h				\
	drivers/mpi/driver_mpi_source.h				\
	drivers/mpi/driver_mpi_sink.h				\
	drivers/disk/driver_disk.h				\
	debug/traces/starpu_fxt.h				\
	profiling/bound.h					\
	profiling/profiling.h					\
	util/openmp_runtime_support.h				\
	util/starpu_clusters_create.h				\
	util/starpu_task_insert_utils.h				\
	util/starpu_data_cpy.h					\
	starpu_parameters.h					\
	sched_policies/prio_deque.h				\
	sched_policies/sched_component.h

libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES = 		\
	common/barrier.c					\
	common/barrier_counter.c				\
	common/bitmap.c						\
	common/hash.c 						\
	common/rwlock.c						\
	common/starpu_spinlock.c				\
	common/timing.c						\
	common/fxt.c						\
	common/utils.c						\
	common/thread.c						\
	common/rbtree.c						\
	common/graph.c						\
	common/inlines.c					\
	core/jobs.c						\
	core/task.c						\
	core/task_bundle.c					\
	core/tree.c						\
	core/drivers.c						\
	core/workers.c						\
	core/combined_workers.c					\
	core/topology.c						\
	core/disk.c						\
	core/debug.c						\
	core/errorcheck.c					\
	core/progress_hook.c					\
	core/idle_hook.c                                        \
	core/dependencies/cg.c					\
	core/dependencies/dependencies.c			\
	core/dependencies/implicit_data_deps.c			\
	core/dependencies/tags.c				\
	core/dependencies/task_deps.c				\
	core/dependencies/data_concurrency.c			\
	core/dependencies/data_arbiter_concurrency.c		\
	core/disk_ops/disk_stdio.c				\
	core/disk_ops/disk_unistd.c                             \
	core/disk_ops/unistd/disk_unistd_global.c		\
	core/perfmodel/perfmodel_history.c			\
	core/perfmodel/perfmodel_bus.c				\
	core/perfmodel/perfmodel.c				\
	core/perfmodel/perfmodel_print.c			\
	core/perfmodel/perfmodel_nan.c				\
	core/perfmodel/regression.c				\
	core/perfmodel/multiple_regression.c			\
	core/sched_policy.c					\
	core/simgrid.c						\
	core/simgrid_cpp.cpp					\
	core/sched_ctx.c					\
	core/sched_ctx_list.c					\
	core/parallel_task.c					\
	core/detect_combined_workers.c				\
	sched_policies/eager_central_policy.c			\
	sched_policies/eager_central_priority_policy.c		\
	sched_policies/work_stealing_policy.c			\
	sched_policies/deque_modeling_policy_data_aware.c	\
	sched_policies/random_policy.c				\
	sched_policies/fifo_queues.c				\
	sched_policies/parallel_heft.c				\
	sched_policies/parallel_eager.c				\
	sched_policies/heteroprio.c				\
	sched_policies/graph_test_policy.c			\
	drivers/driver_common/driver_common.c			\
	drivers/disk/driver_disk.c				\
	datawizard/node_ops.c					\
	datawizard/memory_nodes.c				\
	datawizard/write_back.c					\
	datawizard/coherency.c					\
	datawizard/data_request.c				\
	datawizard/datawizard.c					\
	datawizard/copy_driver.c				\
	datawizard/filters.c					\
	datawizard/sort_data_handles.c				\
	datawizard/malloc.c					\
	datawizard/memory_manager.c				\
	datawizard/memalloc.c					\
	datawizard/memstats.c					\
	datawizard/footprint.c					\
	datawizard/datastats.c					\
	datawizard/user_interactions.c				\
	datawizard/reduction.c					\
	datawizard/interfaces/data_interface.c			\
	datawizard/interfaces/bcsr_interface.c			\
	datawizard/interfaces/coo_interface.c                   \
	datawizard/interfaces/csr_interface.c			\
	datawizard/interfaces/matrix_filters.c			\
	datawizard/interfaces/matrix_interface.c		\
	datawizard/interfaces/block_filters.c			\
	datawizard/interfaces/block_interface.c			\
	datawizard/interfaces/vector_interface.c		\
	datawizard/interfaces/bcsr_filters.c			\
	datawizard/interfaces/csr_filters.c			\
	datawizard/interfaces/vector_filters.c			\
	datawizard/interfaces/variable_interface.c		\
	datawizard/interfaces/void_interface.c			\
	datawizard/interfaces/multiformat_interface.c           \
	util/execute_on_all.c					\
	util/starpu_create_sync_task.c				\
	util/file.c						\
	util/fstarpu.c						\
	util/misc.c						\
	util/openmp_runtime_support.c				\
	util/openmp_runtime_support_environment.c		\
	util/openmp_runtime_support_omp_api.c			\
	util/starpu_data_cpy.c					\
	util/starpu_task_insert.c				\
	util/starpu_task_insert_utils.c				\
	debug/traces/starpu_fxt.c				\
	debug/traces/starpu_fxt_mpi.c				\
	debug/traces/starpu_fxt_dag.c				\
	debug/traces/starpu_paje.c				\
	debug/traces/anim.c					\
	debug/latency.c						\
	debug/structures_size.c					\
	profiling/profiling.c					\
	profiling/bound.c					\
	profiling/profiling_helpers.c				\
	worker_collection/worker_list.c				\
	worker_collection/worker_tree.c				\
	sched_policies/component_worker.c				\
	sched_policies/component_sched.c				\
	sched_policies/component_fifo.c 				\
	sched_policies/prio_deque.c				\
	sched_policies/helper_mct.c				\
	sched_policies/component_prio.c 				\
	sched_policies/component_random.c				\
	sched_policies/component_eager.c				\
	sched_policies/component_eager_prio.c				\
	sched_policies/component_eager_calibration.c				\
	sched_policies/component_mct.c				\
	sched_policies/component_heft.c				\
	sched_policies/component_heteroprio.c				\
	sched_policies/component_best_implementation.c		\
	sched_policies/component_perfmodel_select.c				\
	sched_policies/component_composed.c				\
	sched_policies/component_work_stealing.c				\
	sched_policies/component_stage.c				\
	sched_policies/component_userchoice.c				\
	sched_policies/modular_eager.c				\
	sched_policies/modular_eager_prio.c				\
	sched_policies/modular_eager_prefetching.c				\
	sched_policies/modular_gemm.c				\
	sched_policies/modular_prio.c				\
	sched_policies/modular_prio_prefetching.c				\
	sched_policies/modular_random.c				\
	sched_policies/modular_parallel_random.c		\
	sched_policies/modular_random_prefetching.c			\
	sched_policies/modular_parallel_heft.c			\
	sched_policies/modular_heft.c				\
	sched_policies/modular_heft_prio.c			\
	sched_policies/modular_heteroprio.c			\
	sched_policies/modular_heteroprio_heft.c		\
	sched_policies/modular_heft2.c				\
	sched_policies/modular_ws.c				\
	sched_policies/modular_ez.c


if STARPU_HAVE_LEVELDB
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += core/disk_ops/disk_leveldb.cpp
endif

if STARPU_HAVE_HDF5
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += core/disk_ops/disk_hdf5.c
endif

libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/cpu/driver_cpu.c

if STARPU_USE_CUDA
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/cuda/driver_cuda.c
else
if STARPU_SIMGRID
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/cuda/driver_cuda.c
endif
endif

libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/cuda/starpu_cublas.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/cuda/starpu_cusparse.c

if STARPU_USE_OPENCL
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/opencl/driver_opencl.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/opencl/driver_opencl_utils.c
else
if STARPU_SIMGRID
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/opencl/driver_opencl.c
endif
endif

if STARPU_LINUX_SYS
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += core/disk_ops/disk_unistd_o_direct.c
endif


if STARPU_HAVE_HWLOC
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += \
	sched_policies/scheduler_maker.c			\
	sched_policies/hierarchical_heft.c
if STARPU_HWLOC_HAVE_TOPOLOGY_DUP
if STARPU_HAVE_OPENMP
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += util/starpu_clusters_create.c
endif
endif
endif


#########################################
#										#
#        Generic MP compilation			#
#										#
#########################################

if STARPU_USE_MP
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mp_common/mp_common.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mp_common/source_common.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mp_common/sink_common.c
endif

#########################################
#										#
#	     MIC compilation				#
#										#
#########################################

if STARPU_USE_MIC
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mic/driver_mic_common.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mic/driver_mic_source.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mic/driver_mic_sink.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mic/driver_mic_utils.c
endif

#########################################
#                                       #
#     MPI Master/Slave compilation      #
#                                       #
#########################################

if STARPU_USE_MPI_MASTER_SLAVE
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mpi/driver_mpi_common.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mpi/driver_mpi_source.c
libstarpu_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += drivers/mpi/driver_mpi_sink.c
endif


#########################################

# If some external references appear (U), it means the corresponding .c file has
# only included <starpu.h> and not the internal src/ header which contains the
# static inline definition
dist-hook:
	failed=0 ; \
	look=""; \
	for i in $$( $(GREP) "static inline" $$(find $(srcdir) -name \*.h) | $(SED) -e 's/.*static inline //g' | $(GREP) -v ENAME\#\# | $(SED) -e 's/[^(]* \(\|\*\)\([^ (]*\)(.*/\2/' | $(GREP) -v _starpu_spin_init) ; do \
		if [ -z "$$look" ] ; then \
			look="$$i" ; \
		else \
			look="$$look\|$$i" ; \
		fi ; \
	done ; \
	echo "$$look" ; \
	for j in $(shell find . -name \*.o) ; do \
		nm $$j | $(GREP) -e "U \($$look\)$$" && { echo $$j ; failed=1 ; } ; \
	done ; \
	[ $$failed == 0 ]
