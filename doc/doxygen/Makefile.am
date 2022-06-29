# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

DOX_DIR = $(top_builddir)/doc/doxygen
DOX_CONFIG = $(top_srcdir)/doc/doxygen/doxygen.cfg

DOX_MAIN_DIR = doxygen
DOX_HTML_DIR = html
DOX_LATEX_DIR = latex
DOX_PDF = starpu.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = starpu_config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	chapters/000_introduction.doxy		\
	chapters/101_building.doxy		\
	chapters/110_basic_examples.doxy		\
	chapters/201_advanced_examples.doxy		\
	chapters/210_check_list_performance.doxy		\
	chapters/220_debugging_tools.doxy		\
	chapters/230_online_performance_tools.doxy		\
	chapters/240_offline_performance_tools.doxy		\
	chapters/301_tasks.doxy		\
	chapters/310_data_management.doxy		\
	chapters/320_scheduling.doxy		\
	chapters/330_scheduling_contexts.doxy		\
	chapters/340_scheduling_context_hypervisor.doxy		\
	chapters/350_scheduling_policy_definition.doxy		\
	chapters/390_faq.doxy		\
	chapters/401_out_of_core.doxy		\
	chapters/410_mpi_support.doxy		\
	chapters/412_tcpip_support.doxy		\
	chapters/415_fault_tolerance.doxy	\
	chapters/420_fft_support.doxy		\
	chapters/440_max_fpga_support.doxy		\
	chapters/450_native_fortran_support.doxy		\
	chapters/460_socl_opencl_extensions.doxy		\
	chapters/470_simgrid.doxy		\
	chapters/480_openmp_runtime_support.doxy		\
	chapters/490_clustering_a_machine.doxy		\
	chapters/494_bubble.doxy		\
	chapters/495_interoperability.doxy		\
	chapters/496_transactions.doxy		\
	chapters/497_eclipse_plugin.doxy 		\
	chapters/498_java.doxy 		\
	chapters/500_python.doxy		\
	chapters/501_environment_variables.doxy		\
	chapters/510_configure_options.doxy		\
	chapters/520_files.doxy		\
	chapters/601_scaling_vector_example.doxy		\
	chapters/610_fdl_1_3.doxy		\
	chapters/code/forkmode.c \
	chapters/code/multiformat.c \
	chapters/code/complex.c \
	chapters/code/simgrid.c \
	chapters/code/vector_scal_c.c \
	chapters/code/vector_scal_cpu.c \
	chapters/code/vector_scal_cuda.c \
	chapters/code/vector_scal_opencl.c \
	chapters/code/vector_scal_opencl_codelet.cl \
	chapters/code/disk_copy.c \
	chapters/code/disk_compute.c \
	chapters/code/nf_initexit.f90 \
	chapters/api/bubble_support.doxy \
	chapters/api/fft_support.doxy \
	chapters/api/threads.doxy

images = 	\
	chapters/images/data_trace.png \
	chapters/images/distrib_data.png \
	chapters/images/distrib_data_histo.png \
	chapters/images/paje_draw_histogram.png \
	chapters/images/parallel_worker2.png \
	chapters/images/runtime-par.png \
	chapters/images/starpu_non_linear_memset_regression_based.png \
	chapters/images/starpu_non_linear_memset_regression_based_2.png \
	chapters/images/starpu_starpu_slu_lu_model_11.png \
	chapters/images/starpu_chol_model_11_type.png \
	chapters/images/tasks_size_overhead.png \
	chapters/images/temanejo.png		\
	chapters/images/eclipse_installer.png	\
	chapters/images/eclipse_install_cdt.png	\
	chapters/images/eclipse_hello_build.png	\
	chapters/images/eclipse_hello_run.png	\
	chapters/images/eclipse_hello_fxt.png	\
	chapters/images/eclipse_hello_graph.png	\
	chapters/images/eclipse_hello_vite.png	\
	chapters/images/eclipse_hello_svg_graph.png	\
	chapters/images/eclipse_hello_plugin.png	\
	chapters/images/eclipse_hello_paje_trace.png	\
	chapters/images/eclipse_hello_hgraph.png	\
	chapters/images/eclipse_install_pde.png		\
	chapters/images/starpu_gflops_non_linear_memset_regression_based_energy.png		\
	chapters/images/starpu_log_arr.png		\
	chapters/images/starpu_log_list.png		\
	chapters/images/starpu_non_linear_memset_regression_based_energy.png		\
	chapters/images/starpu_power_non_linear_memset_regression_based.png		\
	chapters/images/starvz_visu.png		\
	chapters/images/starvz_visu_r.png		\
	chapters/images/trace_bw_heatmap.png		\
	chapters/images/trace_recv_use.png		\
	chapters/images/trace_send_use.png		\
	chapters/images/trace_volume_heatmap.png	\
	chapters/images/starpupy_handle_func_perf_pickle.png  \
	chapters/images/starpupy_handle_perf_pickle.png	\
	chapters/images/starpupy_handle_func_perf.png \
	chapters/images/starpupy_handle_perf.png		\
	chapters/images/tasks_size_overhead_py_fut_pickle.png	\
	chapters/images/tasks_size_overhead_py_futur.png	\
	chapters/images/tasks_size_overhead_py_handle_pickle.png \
	chapters/images/tasks_size_overhead_py_handle.png	\
	chapters/images/tasks_size_overhead_py_none.png		\
	chapters/images/tasks_size_overhead_py_noret_pickle.png

if STARPU_BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html				\
	$(top_srcdir)/include/starpu.h			\
	$(top_srcdir)/include/starpu_bitmap.h		\
	$(top_srcdir)/include/starpu_bound.h		\
	$(top_srcdir)/include/starpu_clusters.h		\
	$(top_srcdir)/include/starpu_cublas.h		\
	$(top_srcdir)/include/starpu_cublas_v2.h	\
	$(top_srcdir)/include/starpu_cusparse.h		\
	$(top_srcdir)/include/starpu_cuda.h		\
	$(top_srcdir)/include/starpu_data_filters.h	\
	$(top_srcdir)/include/starpu_data.h		\
	$(top_srcdir)/include/starpu_data_interfaces.h	\
	$(top_srcdir)/include/starpu_deprecated_api.h	\
	$(top_srcdir)/include/starpu_disk.h		\
	$(top_srcdir)/include/starpu_driver.h		\
	$(top_srcdir)/include/starpu_expert.h		\
	$(top_srcdir)/include/starpu_fxt.h		\
	$(top_srcdir)/include/starpu_hash.h		\
	$(top_srcdir)/include/starpu_helper.h		\
	$(top_srcdir)/include/starpu_hip.h		\
	$(top_srcdir)/include/starpu_max_fpga.h		\
	$(top_srcdir)/include/starpu_mod.f90		\
	$(top_srcdir)/include/starpu_opencl.h		\
	$(top_srcdir)/include/starpu_openmp.h		\
	$(top_srcdir)/include/starpu_perf_monitoring.h	\
	$(top_srcdir)/include/starpu_perf_steering.h	\
	$(top_srcdir)/include/starpu_perfmodel.h	\
	$(top_srcdir)/include/starpu_profiling.h	\
	$(top_srcdir)/include/starpu_rand.h		\
	$(top_srcdir)/include/starpu_sched_component.h	\
	$(top_srcdir)/include/starpu_sched_ctx.h	\
	$(top_srcdir)/include/starpu_sched_ctx_hypervisor.h	\
	$(top_srcdir)/include/starpu_scheduler.h	\
	$(top_srcdir)/include/starpu_simgrid_wrap.h	\
	$(top_srcdir)/include/starpu_sink.h		\
	$(top_srcdir)/include/starpu_stdlib.h		\
	$(top_srcdir)/include/starpu_task_bundle.h	\
	$(top_srcdir)/include/starpu_task_dep.h		\
	$(top_srcdir)/include/starpu_task.h		\
	$(top_srcdir)/include/starpu_task_list.h	\
	$(top_srcdir)/include/starpu_task_util.h	\
	$(top_srcdir)/include/starpu_thread.h		\
	$(top_srcdir)/include/starpu_thread_util.h	\
	$(top_srcdir)/include/starpu_tree.h		\
	$(top_srcdir)/include/starpu_util.h		\
	$(top_srcdir)/include/starpu_worker.h		\
	$(top_srcdir)/include/fstarpu_mod.f90		\
	$(top_srcdir)/include/schedulers/starpu_heteroprio.h	\
	$(top_srcdir)/starpufft/include/starpufft.h 	\
	$(top_srcdir)/mpi/include/starpu_mpi.h 		\
	$(top_srcdir)/mpi/include/starpu_mpi_ft.h 	\
	$(top_srcdir)/mpi/include/starpu_mpi_lb.h	\
	$(top_srcdir)/mpi/include/fstarpu_mpi_mod.f90		\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor.h 		\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_config.h 	\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_lp.h		\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_monitoring.h	\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_policy.h	\
	$(top_srcdir)/starpurm/include/starpurm.h			\
	$(top_srcdir)/include/schedulers/starpu_scheduler_toolbox.h
endif

