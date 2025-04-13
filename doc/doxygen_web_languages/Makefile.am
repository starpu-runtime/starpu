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

DOX_DIR = $(top_builddir)/doc/doxygen_web_languages
DOX_CONFIG = $(top_srcdir)/doc/doxygen.cfg

DOX_MAIN_DIR = doxygen_web_languages
DOX_HTML_DIR = html_web_languages
DOX_LATEX_DIR = latex
DOX_PDF = starpu_web_languages.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = starpu_config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	../doxygen/chapters/foreword.doxy		\
	../doxygen/chapters/starpu_languages/languages_intro.doxy 		\
	../doxygen/chapters/starpu_languages/native_fortran_support.doxy		\
	../doxygen/chapters/starpu_languages/java.doxy 		\
	../doxygen/chapters/starpu_languages/python.doxy		\
	../doxygen/chapters/starpu_languages/openmp_runtime_support.doxy		\
	../doxygen/chapters/starpu_languages/code/nf_initexit.f90 \
	../doxygen/chapters/starpu_languages/code/java_starpu.java	\
	../doxygen/chapters/starpu_languages/code/java_spark.java \
	../doxygen/chapters/files.doxy		\
	../doxygen/chapters/fdl_1_3.doxy

images = 	\
	../doxygen/chapters/starpu_languages/images/starpu_log_arr.png \
	../doxygen/chapters/starpu_languages/images/starpu_log_list.png \
	../doxygen/chapters/starpu_languages/images/starpupy_func_perf.png \
	../doxygen/chapters/starpu_languages/images/starpupy_handle_func_perf_pickle.png \
	../doxygen/chapters/starpu_languages/images/starpupy_handle_func_perf.png \
	../doxygen/chapters/starpu_languages/images/starpupy_handle_perf_pickle.png \
	../doxygen/chapters/starpu_languages/images/starpupy_handle_perf.png \
	../doxygen/chapters/starpu_languages/images/starpupy_perf.png \
	../doxygen/chapters/starpu_languages/images/tasks_size_overhead_py_fut_pickle.png \
	../doxygen/chapters/starpu_languages/images/tasks_size_overhead_py_futur.png \
	../doxygen/chapters/starpu_languages/images/tasks_size_overhead_py_handle_pickle.png \
	../doxygen/chapters/starpu_languages/images/tasks_size_overhead_py_handle.png \
	../doxygen/chapters/starpu_languages/images/tasks_size_overhead_py_none.png \
	../doxygen/chapters/starpu_languages/images/tasks_size_overhead_py_noret_pickle.png

if STARPU_BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html
endif

