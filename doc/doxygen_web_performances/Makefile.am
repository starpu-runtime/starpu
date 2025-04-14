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

DOX_DIR = $(top_builddir)/doc/doxygen_web_performances
DOX_CONFIG = $(top_srcdir)/doc/doxygen.cfg

DOX_MAIN_DIR = doxygen_web_performances
DOX_HTML_DIR = html_web_performances
DOX_LATEX_DIR = latex
DOX_PDF = starpu_web_performances.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = starpu_config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	../doxygen/chapters/foreword.doxy		\
	../doxygen/chapters/starpu_performances/performances_intro.doxy		\
	../doxygen/chapters/starpu_performances/benchmarking_starpu.doxy		\
	../doxygen/chapters/starpu_performances/online_performance_tools.doxy		\
	../doxygen/../doxygen/chapters/starpu_performances/offline_performance_tools.doxy		\
	../doxygen/chapters/files.doxy		\
	../doxygen/chapters/fdl_1_3.doxy

images = 	\
	../doxygen/chapters/starpu_performances/images/data_trace.png \
	../doxygen/chapters/starpu_performances/images/distrib_data_histo.png \
	../doxygen/chapters/starpu_performances/images/distrib_data.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_build.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_fxt.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_graph.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_hgraph.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_paje_trace.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_plugin.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_run.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_svg_graph.png \
	../doxygen/chapters/starpu_performances/images/eclipse_hello_vite.png \
	../doxygen/chapters/starpu_performances/images/eclipse_install_cdt.png \
	../doxygen/chapters/starpu_performances/images/eclipse_installer.png \
	../doxygen/chapters/starpu_performances/images/eclipse_install_pde.png \
	../doxygen/chapters/starpu_performances/images/paje_draw_histogram.png \
	../doxygen/chapters/starpu_performances/images/starpu_chol_model_11_type.png \
	../doxygen/chapters/starpu_performances/images/starpu_gflops_non_linear_memset_regression_based_energy.png \
	../doxygen/chapters/starpu_performances/images/starpu_non_linear_memset_regression_based_2.png \
	../doxygen/chapters/starpu_performances/images/starpu_non_linear_memset_regression_based_energy.png \
	../doxygen/chapters/starpu_performances/images/starpu_non_linear_memset_regression_based.png \
	../doxygen/chapters/starpu_performances/images/starpu_power_non_linear_memset_regression_based.png \
	../doxygen/chapters/starpu_performances/images/starvz_visu.png \
	../doxygen/chapters/starpu_performances/images/starvz_visu_r.png \
	../doxygen/chapters/starpu_performances/images/trace_recv_use.png \
	../doxygen/chapters/starpu_performances/images/trace_send_use.png \
	../doxygen/chapters/starpu_performances/images/vite_bottom_curve.png \
	../doxygen/chapters/starpu_performances/images/vite_gflops.png \
	../doxygen/chapters/starpu_performances/images/vite_gpu_curves.png \
	../doxygen/chapters/starpu_performances/images/vite_gpu_memory.png \
	../doxygen/chapters/starpu_performances/images/vite_gpu_memory_select.png \
	../doxygen/chapters/starpu_performances/images/vite_no_arrow.png \
	../doxygen/chapters/starpu_performances/images/vite_open.png \
	../doxygen/chapters/starpu_performances/images/vite_preferences.png \
	../doxygen/chapters/starpu_performances/images/vite_task_state.png \
	../doxygen/chapters/starpu_performances/images/vite_top_curve.png \
	../doxygen/chapters/starpu_performances/images/vite_top_curve_select.png \
	../doxygen/chapters/starpu_performances/images/vite_trans_arrow.png \
	../doxygen/chapters/starpu_performances/images/vite_worker_state.png \
	../doxygen/chapters/starpu_performances/images/vite_zoom_horiz.png \
	../doxygen/chapters/starpu_performances/images/vite_zoom_horiz_result.png \
	../doxygen/chapters/starpu_performances/images/vite_zoom.png \
	../doxygen/chapters/starpu_performances/images/vite_zoom_result.png \
	../doxygen/chapters/starpu_performances/images/taskgraph.0.dot \
	../doxygen/chapters/starpu_performances/images/taskgraph.0.png \
	../doxygen/chapters/starpu_performances/images/traces_tasks.png \
	../doxygen/chapters/starpu_performances/images/traces_threads.png \
	../doxygen/chapters/starpu_performances/images/tasks.json.gz \
	../doxygen/chapters/starpu_performances/images/threads.json.gz

if STARPU_BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html
endif

