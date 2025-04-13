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

DOX_DIR = $(top_builddir)/doc/doxygen_web_installation
DOX_CONFIG = $(top_srcdir)/doc/doxygen.cfg

DOX_MAIN_DIR = doxygen_web_installation
DOX_HTML_DIR = html_web_installation
DOX_LATEX_DIR = latex
DOX_PDF = starpu_web_installation.pdf
DOX_TAG = starpu.tag
DOX_STARPU_CONFIG = starpu_config.h

include $(top_srcdir)/doc/doxy.mk

chapters =	\
	../doxygen/chapters/foreword.doxy		\
	../doxygen/chapters/starpu_installation/installation_intro.doxy		\
	../doxygen/chapters/starpu_installation/environment_variables.doxy		\
	../doxygen/chapters/starpu_installation/building.doxy		\
	../doxygen/chapters/starpu_installation/configure_options.doxy		\
	../doxygen/chapters/starpu_installation/configuration_and_initialization.doxy		\
	../doxygen/chapters/files.doxy		\
	../doxygen/chapters/fdl_1_3.doxy

images =

if STARPU_BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html
endif

