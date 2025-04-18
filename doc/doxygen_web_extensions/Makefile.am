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

DOX_DIR = $(top_builddir)/doc/doxygen_web_extensions
DOX_CONFIG = $(top_srcdir)/doc/doxygen.cfg

DOX_MAIN_DIR = doxygen_web_extensions
DOX_HTML_DIR = html_web_extensions
DOX_LATEX_DIR = latex
DOX_PDF = starpu_web_extensions.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = starpu_config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	../doxygen/chapters/foreword.doxy		\
	../doxygen/chapters/starpu_extensions/extensions_intro.doxy		\
	../doxygen/chapters/starpu_extensions/advanced_tasks.doxy		\
	../doxygen/chapters/starpu_extensions/advanced_data_management.doxy		\
	../doxygen/chapters/starpu_extensions/helpers.doxy		\
	../doxygen/chapters/starpu_extensions/debugging_tools.doxy		\
	../doxygen/chapters/starpu_extensions/advanced_scheduling.doxy		\
	../doxygen/chapters/starpu_extensions/scheduling_contexts.doxy		\
	../doxygen/chapters/starpu_extensions/scheduling_context_hypervisor.doxy		\
	../doxygen/chapters/starpu_extensions/cuda_support.doxy		\
	../doxygen/chapters/starpu_extensions/opencl_support.doxy		\
	../doxygen/chapters/starpu_extensions/max_fpga_support.doxy		\
	../doxygen/chapters/starpu_extensions/out_of_core.doxy		\
	../doxygen/chapters/starpu_extensions/mpi_support.doxy		\
	../doxygen/chapters/starpu_extensions/tcpip_support.doxy		\
	../doxygen/chapters/starpu_extensions/transactions.doxy		\
	../doxygen/chapters/starpu_extensions/fault_tolerance.doxy	\
	../doxygen/chapters/starpu_extensions/fft_support.doxy		\
	../doxygen/chapters/starpu_extensions/socl_opencl_extensions.doxy		\
	../doxygen/chapters/starpu_extensions/recursive_tasks.doxy		\
	../doxygen/chapters/starpu_extensions/parallel_worker.doxy		\
	../doxygen/chapters/starpu_extensions/darts.doxy		\
	../doxygen/chapters/starpu_extensions/interoperability.doxy		\
	../doxygen/chapters/starpu_extensions/scheduling_policy_definition.doxy		\
	../doxygen/chapters/starpu_extensions/simgrid.doxy		\
	../doxygen/chapters/starpu_extensions/code/complex.c \
	../doxygen/chapters/starpu_extensions/code/disk_compute.c \
	../doxygen/chapters/starpu_extensions/code/disk_copy.c \
	../doxygen/chapters/starpu_extensions/code/forkmode.c \
	../doxygen/chapters/starpu_extensions/code/multiformat.c \
	../doxygen/chapters/starpu_extensions/code/simgrid.c \
	../doxygen/chapters/files.doxy		\
	../doxygen/chapters/fdl_1_3.doxy

images = 	\
	../doxygen/chapters/starpu_extensions/images/parallel_worker2.png \
	../doxygen/chapters/starpu_extensions/images/runtime-par.png \
	../doxygen/chapters/starpu_extensions/images/temanejo.png		\
	../doxygen/chapters/starpu_extensions/images/trace_bw_heatmap.png		\
	../doxygen/chapters/starpu_extensions/images/trace_volume_heatmap.png

if STARPU_BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html
endif

