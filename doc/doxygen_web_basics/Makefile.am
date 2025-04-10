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

DOX_DIR = $(top_builddir)/doc/doxygen_web_basics
DOX_CONFIG = $(top_srcdir)/doc/doxygen.cfg

DOX_MAIN_DIR = doxygen_web_basics
DOX_HTML_DIR = html_web_basics
DOX_LATEX_DIR = latex
DOX_PDF = starpu_web_basics.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = starpu_config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	../doxygen/chapters/foreword.doxy		\
	../doxygen/chapters/starpu_basics/basics_intro.doxy		\
	../doxygen/chapters/starpu_basics/starpu_applications.doxy		\
	../doxygen/chapters/starpu_basics/basic_examples.doxy		\
	../doxygen/chapters/starpu_basics/scaling_vector_example.doxy		\
	../doxygen/chapters/starpu_basics/tasks.doxy		\
	../doxygen/chapters/starpu_basics/data_management.doxy		\
	../doxygen/chapters/starpu_basics/scheduling.doxy		\
	../doxygen/chapters/starpu_basics/examples_sources.doxy		\
	../doxygen/chapters/starpu_basics/code/basics_vector_scal_c.c \
	../doxygen/chapters/starpu_basics/code/basics_vector_scal_cpu.c \
	../doxygen/chapters/starpu_basics/code/basics_vector_scal_cuda.c \
	../doxygen/chapters/starpu_basics/code/basics_vector_scal_opencl.c \
	../doxygen/chapters/starpu_basics/code/basics_vector_scal_opencl_codelet.cl \
	../doxygen/chapters/files.doxy		\
	../doxygen/chapters/fdl_1_3.doxy

images = 	\
	../doxygen/chapters/images/arbiter.png \
	../doxygen/chapters/images/data_trace.png \
	../doxygen/chapters/images/distrib_data.png \
	../doxygen/chapters/images/distrib_data_histo.png \
	../doxygen/chapters/images/paje_draw_histogram.png \
	../doxygen/chapters/images/parallel_worker2.png \
	../doxygen/chapters/images/runtime-par.png \
	../doxygen/chapters/images/starpu_non_linear_memset_regression_based.png \
	../doxygen/chapters/images/starpu_non_linear_memset_regression_based_2.png \
	../doxygen/chapters/images/starpu_starpu_slu_lu_model_11.png \
	../doxygen/chapters/images/starpu_chol_model_11_type.png \
	../doxygen/chapters/images/tasks_size_overhead.png \
	../doxygen/chapters/images/temanejo.png		\
	../doxygen/chapters/images/eclipse_installer.png	\
	../doxygen/chapters/images/eclipse_install_cdt.png	\
	../doxygen/chapters/images/eclipse_hello_build.png	\
	../doxygen/chapters/images/eclipse_hello_run.png	\
	../doxygen/chapters/images/eclipse_hello_fxt.png	\
	../doxygen/chapters/images/eclipse_hello_graph.png	\
	../doxygen/chapters/images/eclipse_hello_vite.png	\
	../doxygen/chapters/images/eclipse_hello_svg_graph.png	\
	../doxygen/chapters/images/eclipse_hello_plugin.png	\
	../doxygen/chapters/images/eclipse_hello_paje_trace.png	\
	../doxygen/chapters/images/eclipse_hello_hgraph.png	\
	../doxygen/chapters/images/eclipse_install_pde.png		\
	../doxygen/chapters/images/starpu_gflops_non_linear_memset_regression_based_energy.png		\
	../doxygen/chapters/images/starpu_log_arr.png		\
	../doxygen/chapters/images/starpu_log_list.png		\
	../doxygen/chapters/images/starpu_non_linear_memset_regression_based_energy.png		\
	../doxygen/chapters/images/starpu_power_non_linear_memset_regression_based.png		\
	../doxygen/chapters/images/starvz_visu.png		\
	../doxygen/chapters/images/starvz_visu_r.png		\
	../doxygen/chapters/images/trace_bw_heatmap.png		\
	../doxygen/chapters/images/trace_recv_use.png		\
	../doxygen/chapters/images/trace_send_use.png		\
	../doxygen/chapters/images/trace_volume_heatmap.png	\
	../doxygen/chapters/images/starpupy_handle_func_perf_pickle.png  \
	../doxygen/chapters/images/starpupy_handle_perf_pickle.png	\
	../doxygen/chapters/images/starpupy_handle_func_perf.png \
	../doxygen/chapters/images/starpupy_handle_perf.png		\
	../doxygen/chapters/images/tasks_size_overhead_py_fut_pickle.png	\
	../doxygen/chapters/images/tasks_size_overhead_py_futur.png	\
	../doxygen/chapters/images/tasks_size_overhead_py_handle_pickle.png \
	../doxygen/chapters/images/tasks_size_overhead_py_handle.png	\
	../doxygen/chapters/images/tasks_size_overhead_py_none.png		\
	../doxygen/chapters/images/tasks_size_overhead_py_noret_pickle.png

if STARPU_BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html
endif

