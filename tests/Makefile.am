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
include $(top_srcdir)/make/starpu-tests.mk
include $(top_srcdir)/make/starpu-loader.mk

AM_CFLAGS += $(APP_CFLAGS)
AM_CXXFLAGS += $(APP_CXXFLAGS)
AM_FFLAGS += $(APP_FFLAGS)
AM_FCFLAGS += $(APP_FCFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/src -I$(top_srcdir)/src/ $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(HWLOC_LIBS)
LIBS += $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_HIP_LDFLAGS)
LIBS += $(STARPU_MAX_FPGA_LDFLAGS)

EXTRA_DIST =					\
	helper.h				\
	datawizard/locality.sh			\
	overlap/overlap.sh			\
	datawizard/scal.h			\
	regression/profiles.in			\
	regression/regression.sh.in		\
	regression/profiles.build.only.in	\
	microbenchs/tasks_data_overhead.sh	\
	microbenchs/sync_tasks_data_overhead.sh	\
	microbenchs/async_tasks_data_overhead.sh	\
	microbenchs/tasks_size_overhead.sh	\
	microbenchs/tasks_size_overhead_sched.sh	\
	microbenchs/tasks_size_overhead_scheds.sh	\
	microbenchs/tasks_size_overhead.gp	\
	microbenchs/parallel_dependent_homogeneous_tasks_data.sh	\
	microbenchs/parallel_independent_heterogeneous_tasks_data.sh	\
	microbenchs/parallel_independent_heterogeneous_tasks.sh	\
	microbenchs/parallel_independent_homogeneous_tasks_data.sh	\
	microbenchs/parallel_independent_homogeneous_tasks.sh	\
	microbenchs/bandwidth_scheds.sh		\
	microbenchs/starpu_check.sh		\
	energy/static.sh			\
	energy/dynamic.sh			\
	energy/perfs.gp				\
	datawizard/scratch_opencl_kernel.cl     \
	datawizard/sync_and_notify_data_opencl_codelet.cl\
	coverage/coverage.sh			\
	variable/increment.h \
	variable/increment_opencl_kernel.cl \
	variable/redux_opencl_kernel.cl \
	variable/neutral_opencl_kernel.cl \
	datawizard/interfaces/test_interfaces.h	\
	datawizard/interfaces/bcsr/bcsr_opencl_kernel.cl \
	datawizard/interfaces/coo/coo_opencl_kernel.cl \
	datawizard/interfaces/matrix/matrix_opencl_kernel.cl \
	datawizard/interfaces/variable/variable_opencl_kernel.cl \
	datawizard/interfaces/vector/vector_opencl_kernel.cl \
	datawizard/interfaces/multiformat/multiformat_types.h \
	datawizard/interfaces/multiformat/multiformat_opencl_kernel.cl \
	datawizard/interfaces/multiformat/multiformat_conversion_codelets_kernel.cl \
	datawizard/interfaces/multiformat/advanced/generic.h \
	datawizard/interfaces/csr/csr_opencl_kernel.cl \
	datawizard/interfaces/block/block_opencl_kernel.cl \
	datawizard/interfaces/tensor/tensor_opencl_kernel.cl \
	datawizard/interfaces/ndim/ndim_opencl_kernel.cl \
	perfmodels/opencl_memset_kernel.cl \
	$(MICROBENCHS:=.sh) \
	microbenchs/microbench.sh	\
	memory/memstress2.gp	\
	memory/memstress2.sh	\
	memory/memstress.gp	\
	memory/memstress.sh	\
	maxfpga/LMemLoopbackCpuCode.c	\
	maxfpga/MyTasksManager.maxj	\
	maxfpga/MyTasksMuxManager.maxj	\
	maxfpga/README.txt	\
	maxfpga/StreamFMACpuCode.cpp	\
	maxfpga/Task1.maxj	\
	maxfpga/Task2.maxj	\
	maxfpga/Task3.maxj	\
	datawizard/interfaces/test_interfaces.sh \
	traces/fxt.sh

CLEANFILES = 					\
	*.gcno *.gcda *.linkinfo core starpu_idle_microsec.log *.mod *.png *.output tasks.rec perfs.rec */perfs.rec */*/perfs.rec perfs2.rec fortran90/starpu_mod.f90 bandwidth-*.dat bandwidth.gp bandwidth.eps bandwidth.svg *.csv *.md *.Rmd *.pdf *.html

clean-local:
	-rm -rf overlap/overlap.traces datawizard/locality.traces traces/fxt.traces

BUILT_SOURCES =
SUBDIRS =

examplebindir = $(libdir)/starpu/examples

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA =
endif

testbindir = $(libdir)/starpu/tests

#####################################
# What to install and what to check #
#####################################

TESTS = $(SHELL_TESTS) $(myPROGRAMS)

SHELL_TESTS =

if STARPU_COVERAGE_ENABLED
SHELL_TESTS	+=	coverage/coverage.sh
endif

XFAIL_TESTS	=				\
	errorcheck/invalid_blocking_calls

myPROGRAMS =

myPROGRAMS +=					\
	main/callback				\
	main/bind				\
	main/mkdtemp				\
	main/execute_schedule			\
	main/insert_task_pack			\
	main/insert_task_nullcodelet		\
	main/multithreaded_init			\
	main/empty_task				\
	main/empty_task_chain			\
	main/starpu_worker_exists		\
	main/codelet_null_callback		\
	datawizard/allocate			\
	datawizard/acquire_cb			\
	datawizard/deps				\
	datawizard/user_interaction_implicit	\
	datawizard/interfaces/copy_interfaces	\
	datawizard/numa_overflow		\
	datawizard/locality			\
	datawizard/variable_size		\
	errorcheck/starpu_init_noworker		\
	errorcheck/invalid_tasks		\
	helper/cublas_init			\
	helper/cusparse_init			\
	helper/pinned_memory			\
	helper/execute_on_all			\
	microbenchs/display_structures_size	\
	microbenchs/local_pingpong		\
	overlap/overlap				\
	sched_ctx/sched_ctx_list		\
	sched_ctx/sched_ctx_policy_data		\
	openmp/init_exit_01			\
	openmp/init_exit_02			\
	openmp/environment			\
	openmp/api_01				\
	openmp/parallel_01			\
	openmp/parallel_02			\
	openmp/parallel_03			\
	openmp/parallel_barrier_01		\
	openmp/parallel_master_01		\
	openmp/parallel_master_inline_01	\
	openmp/parallel_single_wait_01		\
	openmp/parallel_single_nowait_01	\
	openmp/parallel_single_inline_01	\
	openmp/parallel_single_copyprivate_01	\
	openmp/parallel_single_copyprivate_inline_01	\
	openmp/parallel_critical_01		\
	openmp/parallel_critical_inline_01	\
	openmp/parallel_critical_named_01	\
	openmp/parallel_critical_named_inline_01\
	openmp/parallel_simple_lock_01		\
	openmp/parallel_nested_lock_01		\
	openmp/parallel_for_01			\
	openmp/parallel_for_02			\
	openmp/parallel_for_ordered_01		\
	openmp/parallel_sections_01		\
	openmp/parallel_sections_combined_01	\
	openmp/task_01				\
	openmp/task_02				\
	openmp/task_03				\
	openmp/taskloop				\
	openmp/taskwait_01			\
	openmp/taskgroup_01			\
	openmp/taskgroup_02			\
	openmp/array_slice_01			\
	openmp/cuda_task_01			\
	perfmodels/value_nan			\
	sched_policies/workerids		\
	sched_policies/help

if STARPU_SIMGRID
myPROGRAMS +=					\
	energy/energy_efficiency		\
	datawizard/simgrid-locality
endif

if !STARPU_SIMGRID
myPROGRAMS +=				\
	main/deprecated_func			\
	main/driver_api/init_run_deinit         \
	main/driver_api/run_driver              \
	main/deploop                            \
	main/display_binding			\
	main/execute_on_a_specific_worker	\
	main/insert_task			\
	main/insert_task_value			\
	main/insert_task_dyn_handles		\
	main/insert_task_array			\
	main/insert_task_many			\
	main/insert_task_where			\
	main/job				\
	main/multithreaded			\
	main/starpu_task_bundle			\
	main/starpu_task_wait_for_all		\
	main/starpu_task_wait			\
	main/static_restartable			\
	main/static_restartable_using_initializer\
	main/static_restartable_tag		\
	main/regenerate				\
	main/regenerate_pipeline		\
	main/restart				\
	main/wait_all_regenerable_tasks		\
	main/subgraph_repeat			\
	main/subgraph_repeat_tag		\
	main/subgraph_repeat_regenerate		\
	main/subgraph_repeat_regenerate_tag	\
	main/subgraph_repeat_regenerate_tag_cycle	\
	main/empty_task_sync_point		\
	main/empty_task_sync_point_tasks	\
	main/tag_wait_api			\
	main/tag_get_task			\
	main/task_wait_api			\
	main/declare_deps_in_callback		\
	main/declare_deps_after_submission	\
	main/declare_deps_after_submission_synchronous	\
	main/get_current_task			\
	main/starpu_init			\
	main/submit				\
	main/const_codelet			\
	main/pause_resume			\
	main/pack				\
	main/get_children_tasks			\
	main/hwloc_cpuset			\
	main/task_end_dep			\
	datawizard/acquire_cb_insert		\
	datawizard/acquire_release		\
	datawizard/acquire_release2		\
	datawizard/acquire_release_to		\
	datawizard/acquire_try			\
	datawizard/bcsr				\
	datawizard/cache			\
	datawizard/commute			\
	datawizard/commute2			\
	datawizard/copy				\
	datawizard/data_implicit_deps		\
	datawizard/data_register		\
	datawizard/scratch			\
	datawizard/scratch_reuse		\
	datawizard/sync_and_notify_data		\
	datawizard/sync_and_notify_data_implicit\
	datawizard/dsm_stress			\
	datawizard/double_parameter		\
	datawizard/write_only_tmp_buffer	\
	datawizard/data_invalidation		\
	datawizard/dining_philosophers		\
	datawizard/manual_reduction		\
	datawizard/readers_and_writers		\
	datawizard/unpartition			\
	datawizard/sync_with_data_with_mem	\
	datawizard/sync_with_data_with_mem_non_blocking\
	datawizard/sync_with_data_with_mem_non_blocking_implicit\
	datawizard/mpi_like			\
	datawizard/mpi_like_async		\
	datawizard/critical_section_with_void_interface\
	datawizard/increment_init		\
	datawizard/increment_redux		\
	datawizard/increment_redux_v2		\
	datawizard/increment_redux_with_args	\
	datawizard/increment_redux_lazy		\
	datawizard/handle_to_pointer		\
	datawizard/lazy_allocation		\
	datawizard/no_unregister		\
	datawizard/noreclaim			\
	datawizard/nowhere			\
	datawizard/interfaces/block/block_interface \
	datawizard/interfaces/bcsr/bcsr_interface \
	datawizard/interfaces/coo/coo_interface \
	datawizard/interfaces/csr/csr_interface \
	datawizard/interfaces/matrix/matrix_interface \
	datawizard/interfaces/multiformat/multiformat_interface \
	datawizard/interfaces/multiformat/advanced/multiformat_cuda_opencl \
	datawizard/interfaces/multiformat/advanced/multiformat_data_release \
	datawizard/interfaces/multiformat/advanced/multiformat_worker \
	datawizard/interfaces/multiformat/advanced/multiformat_handle_conversion \
	datawizard/interfaces/multiformat/advanced/same_handle \
	datawizard/interfaces/tensor/tensor_interface \
	datawizard/interfaces/ndim/ndim_interface \
	datawizard/interfaces/variable/variable_interface    \
	datawizard/interfaces/vector/vector_interface   \
	datawizard/interfaces/void/void_interface \
	datawizard/in_place_partition   	\
	datawizard/partition_dep   		\
	datawizard/partition_lazy		\
	datawizard/partition_init		\
	datawizard/partition_wontuse		\
	datawizard/gpu_register   		\
	datawizard/gpu_ptr_register   		\
	datawizard/variable_parameters		\
	datawizard/wt_host			\
	datawizard/wt_broadcast			\
	datawizard/readonly			\
	datawizard/specific_node		\
	datawizard/task_with_multiple_time_the_same_handle	\
	datawizard/test_arbiter			\
	datawizard/invalidate_pending_requests	\
	datawizard/temporary_partition		\
	datawizard/partitioned_initialization	\
	datawizard/partitioned_acquire		\
	datawizard/temporary_partition_implicit	\
	datawizard/temporary_partition_read	\
	datawizard/redux_acquire		\
	disk/disk_copy				\
	disk/disk_copy_unpack			\
	disk/disk_copy_to_disk			\
	disk/disk_compute			\
	disk/disk_pack				\
	disk/mem_reclaim			\
	errorcheck/invalid_blocking_calls	\
	errorcheck/workers_cpuid		\
	fault-tolerance/retry			\
	helper/starpu_data_cpy			\
	helper/starpu_data_dup_ro		\
	helper/starpu_create_sync_task		\
	microbenchs/async_tasks_overhead	\
	microbenchs/sync_tasks_overhead		\
	microbenchs/tasks_overhead		\
	microbenchs/tasks_size_overhead		\
	microbenchs/prefetch_data_on_node 	\
	microbenchs/redundant_buffer		\
	microbenchs/matrix_as_vector		\
	microbenchs/bandwidth			\
	overlap/gpu_concurrency			\
	parallel_tasks/explicit_combined_worker	\
	parallel_tasks/parallel_kernels		\
	parallel_tasks/parallel_kernels_trivial	\
	parallel_tasks/parallel_kernels_spmd	\
	parallel_tasks/spmd_peager		\
	parallel_tasks/cuda_only		\
	perfmodels/regression_based_memset	\
	perfmodels/regression_based_check	\
	perfmodels/regression_based_multiimpl	\
	perfmodels/regression_based_energy	\
	perfmodels/regression_based_gpu		\
	perfmodels/non_linear_regression_based	\
	perfmodels/feed				\
	perfmodels/user_base			\
	perfmodels/valid_model			\
	perfmodels/path				\
	perfmodels/memory			\
	sched_policies/data_locality            \
	sched_policies/execute_all_tasks        \
	sched_policies/prio        		\
	sched_policies/simple_deps              \
	sched_policies/simple_cpu_gpu_sched	\
	sched_ctx/sched_ctx_hierarchy

noinst_PROGRAMS		+= \
	datawizard/allocate_many_numa_nodes

if STARPU_USE_MAX_FPGA
myPROGRAMS +=				\
        maxfpga/max_fpga_basic_static	\
        maxfpga/max_fpga_advanced_static	\
        maxfpga/max_fpga_dynamic		\
        maxfpga/max_fpga_mux
endif
endif

MICROBENCHS = \
	microbenchs/parallel_independent_homogeneous_tasks      \
	microbenchs/parallel_independent_heterogeneous_tasks    \
	microbenchs/parallel_independent_homogeneous_tasks_data	\
	microbenchs/parallel_independent_heterogeneous_tasks_data \
	microbenchs/parallel_redux_homogeneous_tasks_data \
	microbenchs/parallel_redux_heterogeneous_tasks_data \
	microbenchs/parallel_dependent_homogeneous_tasks_data

if STARPU_HAVE_FC
if !STARPU_SANITIZE
if !STARPU_SIMGRID
myPROGRAMS +=				\
	fortran90/init_01
endif
endif
endif

if !STARPU_SIMGRID
if STARPU_LONG_CHECK
myPROGRAMS +=				\
	main/tag_task_data_deps			\
	datawizard/reclaim
endif
endif

examplebin_PROGRAMS = \
	main/deadlock				\
	microbenchs/async_tasks_overhead	\
	microbenchs/sync_tasks_overhead		\
	microbenchs/tasks_overhead		\
	microbenchs/tasks_size_overhead		\
	microbenchs/local_pingpong
examplebin_SCRIPTS = \
	microbenchs/tasks_data_overhead.sh \
	microbenchs/sync_tasks_data_overhead.sh \
	microbenchs/async_tasks_data_overhead.sh \
	microbenchs/tasks_size_overhead.gp \
	microbenchs/tasks_size_overhead.sh
if !STARPU_SIMGRID
if !STARPU_USE_MPI_MASTER_SLAVE
examplebin_PROGRAMS += \
	microbenchs/bandwidth
SHELL_TESTS += \
	microbenchs/tasks_data_overhead.sh \
	microbenchs/sync_tasks_data_overhead.sh \
	microbenchs/async_tasks_data_overhead.sh \
	microbenchs/tasks_size_overhead_scheds.sh
endif
endif

if STARPU_HAVE_WINDOWS
check_PROGRAMS	=	$(myPROGRAMS)
else
check_PROGRAMS	=	$(LOADER) $(myPROGRAMS)
endif
noinst_PROGRAMS += $(myPROGRAMS)
noinst_PROGRAMS += $(MICROBENCHS)

if STARPU_SIMGRID
SHELL_TESTS += $(MICROBENCHS:=.sh)
endif

SHELL_TESTS += \
	traces/fxt.sh \
	datawizard/locality.sh \
	microbenchs/bandwidth_scheds.sh

if STARPU_USE_FXT
SHELL_TESTS += \
	overlap/overlap.sh
endif

################################
# Simgrid Model Checking tests #
################################

if STARPU_SIMGRID_MC
SUBDIRS += model-checking
endif

#######################
# Source files        #
#######################

datawizard_acquire_release_SOURCES =		\
	datawizard/acquire_release.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_acquire_release_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_acquire_release_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_acquire_release_SOURCES +=		\
	variable/increment_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	variable/increment_opencl_kernel.cl
endif

datawizard_acquire_release2_SOURCES =		\
	datawizard/acquire_release2.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_acquire_release2_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_acquire_release2_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_acquire_release2_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_acquire_release_to_SOURCES =		\
	datawizard/acquire_release_to.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_acquire_release_to_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_acquire_release_to_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_acquire_release_to_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_wt_host_SOURCES =		\
	datawizard/wt_host.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_wt_host_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_wt_host_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_wt_host_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_wt_broadcast_SOURCES =		\
	datawizard/wt_broadcast.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_wt_broadcast_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_wt_broadcast_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_wt_broadcast_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_increment_redux_lazy_SOURCES =		\
	datawizard/increment_redux_lazy.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_increment_redux_lazy_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_increment_redux_lazy_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_increment_redux_lazy_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_increment_redux_SOURCES =		\
	datawizard/increment_redux.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_increment_redux_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_increment_redux_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_increment_redux_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_increment_redux_v2_SOURCES =		\
	datawizard/increment_redux_v2.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_increment_redux_v2_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_increment_redux_v2_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_increment_redux_v2_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_increment_redux_with_args_SOURCES =		\
	datawizard/increment_redux_with_args.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_increment_redux_with_args_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_increment_redux_with_args_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_increment_redux_with_args_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_increment_init_SOURCES =		\
	datawizard/increment_init.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_increment_init_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_increment_init_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_increment_init_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_scratch_SOURCES =			\
	datawizard/scratch.c
if STARPU_USE_CUDA
datawizard_scratch_SOURCES +=		\
	datawizard/scratch_cuda.cu
endif
if STARPU_USE_OPENCL
datawizard_scratch_SOURCES += \
	datawizard/scratch_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/scratch_opencl_kernel.cl
endif

datawizard_mpi_like_SOURCES =		\
	datawizard/mpi_like.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_mpi_like_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_mpi_like_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_mpi_like_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_mpi_like_async_SOURCES =		\
	datawizard/mpi_like_async.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_mpi_like_async_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_mpi_like_async_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_mpi_like_async_SOURCES +=		\
	variable/increment_opencl.c
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

datawizard_in_place_partition_SOURCES =	\
	datawizard/in_place_partition.c	\
	datawizard/scal.c
if STARPU_USE_CUDA
datawizard_in_place_partition_SOURCES +=	\
	datawizard/scal_cuda.cu
endif
if STARPU_USE_OPENCL
datawizard_in_place_partition_SOURCES +=	\
	datawizard/scal_opencl.cl
endif

datawizard_partition_dep_SOURCES =	\
	datawizard/partition_dep.c	\
	datawizard/scal.c
if STARPU_USE_CUDA
datawizard_partition_dep_SOURCES +=	\
	datawizard/scal_cuda.cu
endif
if STARPU_USE_OPENCL
datawizard_partition_dep_SOURCES +=	\
	datawizard/scal_opencl.cl
endif

datawizard_partition_lazy_SOURCES =	\
	datawizard/partition_lazy.c	\
	datawizard/scal.c
if STARPU_USE_CUDA
datawizard_partition_lazy_SOURCES +=	\
	datawizard/scal_cuda.cu
endif
if STARPU_USE_OPENCL
datawizard_partition_lazy_SOURCES +=	\
	datawizard/scal_opencl.cl
endif

datawizard_gpu_register_SOURCES =	\
	datawizard/gpu_register.c	\
	datawizard/scal.c
if STARPU_USE_CUDA
datawizard_gpu_register_SOURCES +=	\
	datawizard/scal_cuda.cu
endif
if STARPU_USE_OPENCL
datawizard_gpu_register_SOURCES +=	\
	datawizard/scal_opencl.cl
endif

datawizard_gpu_ptr_register_SOURCES =	\
	datawizard/gpu_ptr_register.c	\
	datawizard/scal.c
if STARPU_USE_CUDA
datawizard_gpu_ptr_register_SOURCES +=	\
	datawizard/scal_cuda.cu
endif
if STARPU_USE_OPENCL
datawizard_gpu_ptr_register_SOURCES +=	\
	datawizard/scal_opencl.cl
endif

datawizard_specific_node_SOURCES =		\
	datawizard/specific_node.c		\
	variable/increment.c
if STARPU_USE_CUDA
datawizard_specific_node_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
datawizard_specific_node_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
datawizard_specific_node_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_test_arbiter_SOURCES =	\
	datawizard/test_arbiter.cpp

main_starpu_worker_exists_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

main_deprecated_func_CFLAGS = $(AM_CFLAGS) -Wno-deprecated-declarations

main_insert_task_where_SOURCES =		\
	main/insert_task_where.c		\
	variable/increment.c
if STARPU_USE_CUDA
main_insert_task_where_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
main_insert_task_where_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
main_insert_task_where_SOURCES +=		\
	variable/increment_opencl.c
endif

main_subgraph_repeat_SOURCES =		\
	main/subgraph_repeat.c		\
	variable/increment.c
if STARPU_USE_CUDA
main_subgraph_repeat_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
main_subgraph_repeat_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
main_subgraph_repeat_SOURCES +=		\
	variable/increment_opencl.c
endif

main_subgraph_repeat_tag_SOURCES =	\
	main/subgraph_repeat_tag.c	\
	variable/increment.c
if STARPU_USE_CUDA
main_subgraph_repeat_tag_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
main_subgraph_repeat_tag_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
main_subgraph_repeat_tag_SOURCES +=		\
	variable/increment_opencl.c
endif

main_subgraph_repeat_regenerate_SOURCES =		\
	main/subgraph_repeat_regenerate.c		\
	variable/increment.c
if STARPU_USE_CUDA
main_subgraph_repeat_regenerate_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
main_subgraph_repeat_regenerate_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
main_subgraph_repeat_regenerate_SOURCES +=		\
	variable/increment_opencl.c
endif

main_subgraph_repeat_regenerate_tag_SOURCES =	\
	main/subgraph_repeat_regenerate_tag.c	\
	variable/increment.c
if STARPU_USE_CUDA
main_subgraph_repeat_regenerate_tag_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
main_subgraph_repeat_regenerate_tag_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
main_subgraph_repeat_regenerate_tag_SOURCES +=		\
	variable/increment_opencl.c
endif

main_subgraph_repeat_regenerate_tag_cycle_SOURCES =	\
	main/subgraph_repeat_regenerate_tag_cycle.c	\
	variable/increment.c
if STARPU_USE_CUDA
main_subgraph_repeat_regenerate_tag_cycle_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
main_subgraph_repeat_regenerate_tag_cycle_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
main_subgraph_repeat_regenerate_tag_cycle_SOURCES +=		\
	variable/increment_opencl.c
endif


if STARPU_HAVE_FC
fortran90_init_01_SOURCES =	\
	fortran90/starpu_mod.f90	\
	fortran90/init_01.f90
endif

helper_starpu_data_dup_ro_SOURCES =		\
	helper/starpu_data_dup_ro.c	\
	variable/increment.c
if STARPU_USE_CUDA
helper_starpu_data_dup_ro_SOURCES +=		\
	variable/increment_cuda.cu
endif
if STARPU_USE_HIP
helper_starpu_data_dup_ro_SOURCES +=		\
	variable/increment_hip.hip
endif
if STARPU_USE_OPENCL
helper_starpu_data_dup_ro_SOURCES +=		\
	variable/increment_opencl.c
endif

datawizard_interfaces_copy_interfaces_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

datawizard_data_register_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

###################
# Block interface #
###################
datawizard_interfaces_block_block_interface_SOURCES= \
	datawizard/interfaces/test_interfaces.c  \
	datawizard/interfaces/block/block_interface.c

datawizard_interfaces_block_block_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_block_block_interface_SOURCES+= \
	datawizard/interfaces/block/block_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_block_block_interface_SOURCES+= \
	datawizard/interfaces/block/block_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/block/block_opencl_kernel.cl
endif

###################
# Tensor interface #
###################
datawizard_interfaces_tensor_tensor_interface_SOURCES= \
	datawizard/interfaces/test_interfaces.c  \
	datawizard/interfaces/tensor/tensor_interface.c

datawizard_interfaces_tensor_tensor_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_tensor_tensor_interface_SOURCES+= \
	datawizard/interfaces/tensor/tensor_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_tensor_tensor_interface_SOURCES+= \
	datawizard/interfaces/tensor/tensor_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/tensor/tensor_opencl_kernel.cl
endif

###################
# Ndim interface #
###################
datawizard_interfaces_ndim_ndim_interface_SOURCES= \
	datawizard/interfaces/test_interfaces.c  \
	datawizard/interfaces/ndim/ndim_interface.c

datawizard_interfaces_ndim_ndim_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_ndim_ndim_interface_SOURCES+= \
	datawizard/interfaces/ndim/ndim_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_ndim_ndim_interface_SOURCES+= \
	datawizard/interfaces/ndim/ndim_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/ndim/ndim_opencl_kernel.cl
endif

##################
# BSCR interface #
##################
datawizard_interfaces_bcsr_bcsr_interface_SOURCES= \
	datawizard/interfaces/test_interfaces.c \
	datawizard/interfaces/bcsr/bcsr_interface.c

datawizard_interfaces_bcsr_bcsr_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_bcsr_bcsr_interface_SOURCES+= \
	datawizard/interfaces/bcsr/bcsr_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_bcsr_bcsr_interface_SOURCES+= \
	datawizard/interfaces/bcsr/bcsr_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/bcsr/bcsr_opencl_kernel.cl
endif

#################
# COO interface #
#################
datawizard_interfaces_coo_coo_interface_SOURCES= \
	datawizard/interfaces/test_interfaces.c \
	datawizard/interfaces/coo/coo_interface.c

datawizard_interfaces_coo_coo_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_coo_coo_interface_SOURCES+= \
	datawizard/interfaces/coo/coo_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_coo_coo_interface_SOURCES+= \
	datawizard/interfaces/coo/coo_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/coo/coo_opencl_kernel.cl
endif

#################
# CSR interface #
#################
datawizard_interfaces_csr_csr_interface_SOURCES= \
	datawizard/interfaces/test_interfaces.c  \
	datawizard/interfaces/csr/csr_interface.c

datawizard_interfaces_csr_csr_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_csr_csr_interface_SOURCES+= \
	datawizard/interfaces/csr/csr_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_csr_csr_interface_SOURCES+= \
	datawizard/interfaces/csr/csr_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/csr/csr_opencl_kernel.cl
endif


####################
# Vector interface #
####################
datawizard_interfaces_vector_vector_interface_SOURCES =               \
	datawizard/interfaces/vector/vector_interface.c               \
	datawizard/interfaces/test_interfaces.c

datawizard_interfaces_vector_vector_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_vector_vector_interface_SOURCES +=               \
	datawizard/interfaces/vector/vector_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_vector_vector_interface_SOURCES +=               \
	datawizard/interfaces/vector/vector_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/vector/vector_opencl_kernel.cl
endif

####################
# Matrix interface #
####################
datawizard_interfaces_matrix_matrix_interface_SOURCES= \
	datawizard/interfaces/test_interfaces.c        \
	datawizard/interfaces/matrix/matrix_interface.c

datawizard_interfaces_matrix_matrix_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_matrix_matrix_interface_SOURCES+= \
	datawizard/interfaces/matrix/matrix_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_matrix_matrix_interface_SOURCES+= \
	datawizard/interfaces/matrix/matrix_opencl.c
nobase_STARPU_OPENCL_DATA_DATA+= \
	datawizard/interfaces/matrix/matrix_opencl_kernel.cl
endif


#########################
# Multiformat interface #
#########################
datawizard_interfaces_multiformat_multiformat_interface_SOURCES =           \
	datawizard/interfaces/test_interfaces.c                             \
	datawizard/interfaces/multiformat/multiformat_interface.c           \
	datawizard/interfaces/multiformat/multiformat_conversion_codelets.c

datawizard_interfaces_multiformat_multiformat_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_multiformat_multiformat_interface_SOURCES+=                  \
	datawizard/interfaces/multiformat/multiformat_cuda.cu                      \
	datawizard/interfaces/multiformat/multiformat_conversion_codelets_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_multiformat_multiformat_interface_SOURCES+=                  \
	datawizard/interfaces/multiformat/multiformat_opencl.c                     \
	datawizard/interfaces/multiformat/multiformat_conversion_codelets_opencl.c
nobase_STARPU_OPENCL_DATA_DATA +=                                                          \
	datawizard/interfaces/multiformat/multiformat_opencl_kernel.cl                     \
	datawizard/interfaces/multiformat/multiformat_conversion_codelets_kernel.cl
endif

datawizard_interfaces_multiformat_advanced_multiformat_cuda_opencl_SOURCES=\
	datawizard/interfaces/multiformat/advanced/generic.c               \
	datawizard/interfaces/multiformat/advanced/multiformat_cuda_opencl.c

datawizard_interfaces_multiformat_advanced_multiformat_data_release_SOURCES = \
	datawizard/interfaces/multiformat/advanced/generic.c                  \
	datawizard/interfaces/multiformat/advanced/multiformat_data_release.c

datawizard_interfaces_multiformat_advanced_multiformat_worker_SOURCES=\
	datawizard/interfaces/multiformat/advanced/generic.c               \
	datawizard/interfaces/multiformat/advanced/multiformat_worker.c

datawizard_interfaces_multiformat_advanced_multiformat_handle_conversion_SOURCES = \
	datawizard/interfaces/multiformat/advanced/generic.c \
	datawizard/interfaces/multiformat/advanced/multiformat_handle_conversion.c

datawizard_interfaces_multiformat_advanced_same_handle_SOURCES= \
	datawizard/interfaces/multiformat/advanced/generic.c               \
	datawizard/interfaces/multiformat/advanced/same_handle.c


######################
# Variable interface #
######################
datawizard_interfaces_variable_variable_interface_SOURCES=   \
	datawizard/interfaces/test_interfaces.c              \
	datawizard/interfaces/variable/variable_interface.c

datawizard_interfaces_variable_variable_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

if STARPU_USE_CUDA
datawizard_interfaces_variable_variable_interface_SOURCES+= \
	datawizard/interfaces/variable/variable_cuda.cu
endif

if STARPU_USE_OPENCL
datawizard_interfaces_variable_variable_interface_SOURCES+= \
	datawizard/interfaces/variable/variable_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	datawizard/interfaces/variable/variable_opencl_kernel.cl
endif

##################
# Void interface #
##################
datawizard_interfaces_void_void_interface_SOURCES=\
	datawizard/interfaces/test_interfaces.c        \
	datawizard/interfaces/void/void_interface.c

datawizard_interfaces_void_void_interface_CFLAGS = $(AM_CFLAGS) $(FXT_CFLAGS)

overlap_gpu_concurrency_SOURCES=\
	overlap/gpu_concurrency.c
if STARPU_USE_CUDA
overlap_gpu_concurrency_SOURCES+=\
	overlap/long_kernel.cu
endif

perfmodels_regression_based_memset_SOURCES=\
	perfmodels/regression_based_memset.c

perfmodels_regression_based_gpu_SOURCES=\
	perfmodels/regression_based_gpu.c

maxfpga_max_fpga_basic_static_LDADD = $(LDADD) \
       maxfpga/slic_MyTasks.o
maxfpga/max_fpga_basic_static.o: maxfpga/MyTasks.max

maxfpga_max_fpga_advanced_static_LDADD = $(LDADD) \
       maxfpga/slic_MyTasks.o
maxfpga/max_fpga_advanced_static.o: maxfpga/MyTasks.max

maxfpga_max_fpga_dynamic_LDADD = $(LDADD) \
       maxfpga/slic_MyTasks.o
maxfpga/max_fpga_dynamic.o: maxfpga/MyTasks.max

maxfpga_max_fpga_mux_LDADD = $(LDADD) \
       maxfpga/slic_MyTasksMux.o
maxfpga/max_fpga_mux.o: maxfpga/MyTasksMux.max

if STARPU_USE_OPENCL
perfmodels_regression_based_memset_SOURCES+=\
	perfmodels/opencl_memset.c

perfmodels_regression_based_gpu_SOURCES+=\
	perfmodels/opencl_memset.c

nobase_STARPU_OPENCL_DATA_DATA += \
	perfmodels/opencl_memset_kernel.cl
endif

perfmodels_non_linear_regression_based_SOURCES=\
	perfmodels/non_linear_regression_based.c

if STARPU_USE_OPENCL
perfmodels_non_linear_regression_based_SOURCES+=\
	perfmodels/opencl_memset.c
endif

sched_policies_execute_all_tasks_LDFLAGS = $(AM_LDFLAGS) -lm
sched_policies_execute_all_tasks_CFLAGS = $(AM_LDFLAGS) $(FXT_CFLAGS)

# Fortran90 tests

# - link over source file to build our own object
fortran90/starpu_mod.f90:
	@$(MKDIR_P) $(dir $@)
	$(V_ln) $(LN_S) $(abs_top_srcdir)/include/$(notdir $@) $@

if STARPU_HAVE_FC
# - express the creation of .mod along .o
starpu_mod.mod: fortran90/starpu_mod.o

# - list explicit dependences to control proper module files dependencies
fortran90/init_01.o: starpu_mod.mod
endif

# Maxeler compiler
if STARPU_USE_MAX_FPGA
# or DFE for hardware execution
MAX_TARGET ?= DFE_SIM
MAX_DFE = MAX5C_$(MAX_TARGET)

MAXJ=$(wildcard maxfpga/*.maxj)
CLASS=$(MAXJ:.maxj=.class)
$(CLASS) &: $(MAXJ)
	maxjc -1.7 -cp $$MAXCLASSPATH $(dir $<)

%.max: %Manager.class
	CLASSPATH=$$CLASSPATH:. maxJavaRun $(shell dirname $*).$(notdir $*)Manager DFEModel=MAIA maxFileName=$(notdir $*) target=$(MAX_TARGET)
	cp $(notdir $*)_$(MAX_DFE)/results/$(notdir $*).{max,h} $(dir $@)

slic_%.o: %.max
	sliccompile $< $@

CLEANFILES += */*.max */*.class max_fpga/*.h
endif

