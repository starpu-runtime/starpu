# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
DOX_CONFIG = $(top_srcdir)/doc/doxygen.cfg

DOX_MAIN_DIR = doxygen
DOX_HTML_DIR = html
DOX_LATEX_DIR = latex
DOX_PDF = starpu.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = starpu_config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	chapters/foreword.doxy		\
	chapters/starpu_introduction/introduction_intro.doxy		\
	chapters/starpu_introduction/doc_organization.doxy		\
	chapters/starpu_introduction/glossary.doxy		\
	chapters/starpu_installation/installation_intro.doxy		\
	chapters/starpu_installation/environment_variables.doxy		\
	chapters/starpu_installation/building.doxy		\
	chapters/starpu_installation/configure_options.doxy		\
	chapters/starpu_installation/configuration_and_initialization.doxy		\
	chapters/starpu_basics/basics_intro.doxy		\
	chapters/starpu_basics/starpu_applications.doxy		\
	chapters/starpu_basics/basic_examples.doxy		\
	chapters/starpu_basics/scaling_vector_example.doxy		\
	chapters/starpu_basics/tasks.doxy		\
	chapters/starpu_basics/data_management.doxy		\
	chapters/starpu_basics/scheduling.doxy		\
	chapters/starpu_basics/examples_sources.doxy		\
	chapters/starpu_basics/code/basics_vector_scal_c.c \
	chapters/starpu_basics/code/basics_vector_scal_cpu.c \
	chapters/starpu_basics/code/basics_vector_scal_cuda.c \
	chapters/starpu_basics/code/basics_vector_scal_opencl.c \
	chapters/starpu_basics/code/basics_vector_scal_opencl_codelet.cl \
	chapters/starpu_applications/applications_intro.doxy		\
	chapters/starpu_applications/vector_scaling.doxy		\
	chapters/starpu_applications/code/vector_scal_c.c		\
	chapters/starpu_applications/code/vector_scal_c_align.c		\
	chapters/starpu_applications/code/vector_scal_cpu.c		\
	chapters/starpu_applications/code/vector_scal_starpu.c		\
	chapters/starpu_applications/stencil.doxy		\
	chapters/starpu_applications/code/stencil5.c	\
	chapters/starpu_applications/code/stencil5_starpu.c	\
	chapters/starpu_applications/code/stencil5_starpu_mpi.c	\
	chapters/starpu_performances/performances_intro.doxy		\
	chapters/starpu_performances/benchmarking_starpu.doxy		\
	chapters/starpu_performances/online_performance_tools.doxy		\
	chapters/starpu_performances/offline_performance_tools.doxy		\
	chapters/starpu_faq/faq_intro.doxy		\
	chapters/starpu_faq/check_list_performance.doxy		\
	chapters/starpu_faq/faq.doxy		\
	chapters/starpu_languages/languages_intro.doxy 		\
	chapters/starpu_languages/native_fortran_support.doxy		\
	chapters/starpu_languages/java.doxy 		\
	chapters/starpu_languages/python.doxy		\
	chapters/starpu_languages/openmp_runtime_support.doxy		\
	chapters/starpu_languages/code/nf_initexit.f90 \
	chapters/starpu_languages/code/java_starpu.java	\
	chapters/starpu_languages/code/java_spark.java	\
	chapters/starpu_extensions/extensions_intro.doxy		\
	chapters/starpu_extensions/advanced_tasks.doxy		\
	chapters/starpu_extensions/advanced_data_management.doxy		\
	chapters/starpu_extensions/helpers.doxy		\
	chapters/starpu_extensions/debugging_tools.doxy		\
	chapters/starpu_extensions/advanced_scheduling.doxy		\
	chapters/starpu_extensions/scheduling_contexts.doxy		\
	chapters/starpu_extensions/scheduling_context_hypervisor.doxy		\
	chapters/starpu_extensions/cuda_support.doxy		\
	chapters/starpu_extensions/opencl_support.doxy		\
	chapters/starpu_extensions/max_fpga_support.doxy		\
	chapters/starpu_extensions/out_of_core.doxy		\
	chapters/starpu_extensions/mpi_support.doxy		\
	chapters/starpu_extensions/tcpip_support.doxy		\
	chapters/starpu_extensions/transactions.doxy		\
	chapters/starpu_extensions/fault_tolerance.doxy	\
	chapters/starpu_extensions/fft_support.doxy		\
	chapters/starpu_extensions/socl_opencl_extensions.doxy		\
	chapters/starpu_extensions/recursive_tasks.doxy		\
	chapters/starpu_extensions/parallel_worker.doxy		\
	chapters/starpu_extensions/darts.doxy \
	chapters/starpu_extensions/interoperability.doxy		\
	chapters/starpu_extensions/scheduling_policy_definition.doxy		\
	chapters/starpu_extensions/simgrid.doxy		\
	chapters/starpu_extensions/code/complex.c \
	chapters/starpu_extensions/code/disk_compute.c \
	chapters/starpu_extensions/code/disk_copy.c \
	chapters/starpu_extensions/code/forkmode.c \
	chapters/starpu_extensions/code/multiformat.c \
	chapters/starpu_extensions/code/simgrid.c \
	chapters/files.doxy		\
	chapters/fdl_1_3.doxy		\
	chapters/api/fortran_support.doxy \
	chapters/api/recursive_tasks_support.doxy \
	chapters/api/fft_support.doxy \
	chapters/api/threads.doxy

images = 	\
	chapters/starpu_basics/images/arbiter.png \
	chapters/starpu_basics/images/tasks_size_overhead.png \
	chapters/starpu_extensions/images/parallel_worker2.png \
	chapters/starpu_extensions/images/runtime-par.png \
	chapters/starpu_extensions/images/temanejo.png \
	chapters/starpu_extensions/images/trace_bw_heatmap.png \
	chapters/starpu_extensions/images/trace_volume_heatmap.png \
	chapters/starpu_faq/images/starpu_starpu_slu_lu_model_11.png \
	chapters/starpu_languages/images/starpu_log_arr.png \
	chapters/starpu_languages/images/starpu_log_list.png \
	chapters/starpu_languages/images/starpupy_func_perf.png \
	chapters/starpu_languages/images/starpupy_handle_func_perf_pickle.png \
	chapters/starpu_languages/images/starpupy_handle_func_perf.png \
	chapters/starpu_languages/images/starpupy_handle_perf_pickle.png \
	chapters/starpu_languages/images/starpupy_handle_perf.png \
	chapters/starpu_languages/images/starpupy_perf.png \
	chapters/starpu_languages/images/tasks_size_overhead_py_fut_pickle.png \
	chapters/starpu_languages/images/tasks_size_overhead_py_futur.png \
	chapters/starpu_languages/images/tasks_size_overhead_py_handle_pickle.png \
	chapters/starpu_languages/images/tasks_size_overhead_py_handle.png \
	chapters/starpu_languages/images/tasks_size_overhead_py_none.png \
	chapters/starpu_languages/images/tasks_size_overhead_py_noret_pickle.png \
	chapters/starpu_performances/images/data_trace.png \
	chapters/starpu_performances/images/distrib_data_histo.png \
	chapters/starpu_performances/images/distrib_data.png \
	chapters/starpu_performances/images/eclipse_hello_build.png \
	chapters/starpu_performances/images/eclipse_hello_fxt.png \
	chapters/starpu_performances/images/eclipse_hello_graph.png \
	chapters/starpu_performances/images/eclipse_hello_hgraph.png \
	chapters/starpu_performances/images/eclipse_hello_paje_trace.png \
	chapters/starpu_performances/images/eclipse_hello_plugin.png \
	chapters/starpu_performances/images/eclipse_hello_run.png \
	chapters/starpu_performances/images/eclipse_hello_svg_graph.png \
	chapters/starpu_performances/images/eclipse_hello_vite.png \
	chapters/starpu_performances/images/eclipse_install_cdt.png \
	chapters/starpu_performances/images/eclipse_installer.png \
	chapters/starpu_performances/images/eclipse_install_pde.png \
	chapters/starpu_performances/images/paje_draw_histogram.png \
	chapters/starpu_performances/images/starpu_chol_model_11_type.png \
	chapters/starpu_performances/images/starpu_gflops_non_linear_memset_regression_based_energy.png \
	chapters/starpu_performances/images/starpu_non_linear_memset_regression_based_2.png \
	chapters/starpu_performances/images/starpu_non_linear_memset_regression_based_energy.png \
	chapters/starpu_performances/images/starpu_non_linear_memset_regression_based.png \
	chapters/starpu_performances/images/starpu_power_non_linear_memset_regression_based.png \
	chapters/starpu_performances/images/starvz_visu.png \
	chapters/starpu_performances/images/starvz_visu_r.png \
	chapters/starpu_performances/images/trace_recv_use.png \
	chapters/starpu_performances/images/trace_send_use.png \
	chapters/starpu_performances/images/vite_bottom_curve.png \
	chapters/starpu_performances/images/vite_gflops.png \
	chapters/starpu_performances/images/vite_gpu_curves.png \
	chapters/starpu_performances/images/vite_gpu_memory.png \
	chapters/starpu_performances/images/vite_gpu_memory_select.png \
	chapters/starpu_performances/images/vite_no_arrow.png \
	chapters/starpu_performances/images/vite_open.png \
	chapters/starpu_performances/images/vite_preferences.png \
	chapters/starpu_performances/images/vite_task_state.png \
	chapters/starpu_performances/images/vite_top_curve.png \
	chapters/starpu_performances/images/vite_top_curve_select.png \
	chapters/starpu_performances/images/vite_trans_arrow.png \
	chapters/starpu_performances/images/vite_worker_state.png \
	chapters/starpu_performances/images/vite_zoom_horiz.png \
	chapters/starpu_performances/images/vite_zoom_horiz_result.png \
	chapters/starpu_performances/images/vite_zoom.png \
	chapters/starpu_performances/images/vite_zoom_result.png \
	chapters/starpu_performances/images/taskgraph.0.dot \
	chapters/starpu_performances/images/taskgraph.0.png \
	chapters/starpu_performances/images/traces_tasks.png \
	chapters/starpu_performances/images/traces_threads.png \
	chapters/starpu_performances/images/tasks.json.gz \
	chapters/starpu_performances/images/threads.json.gz


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
	$(top_srcdir)/include/starpu_cublas.h		\
	$(top_srcdir)/include/starpu_cublas_v2.h	\
	$(top_srcdir)/include/starpu_cublasLt.h		\
	$(top_srcdir)/include/starpu_cusparse.h		\
	$(top_srcdir)/include/starpu_cuda.h		\
	$(top_srcdir)/include/starpu_cusolver.h		\
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
	$(top_srcdir)/include/starpu_parallel_worker.h	\
	$(top_srcdir)/include/starpu_perf_monitoring.h	\
	$(top_srcdir)/include/starpu_perf_steering.h	\
	$(top_srcdir)/include/starpu_perfmodel.h	\
	$(top_srcdir)/include/starpu_profiling.h	\
	$(top_srcdir)/include/starpu_profiling_tool.h	\
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

