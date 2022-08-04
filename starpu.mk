# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS = $(GLOBAL_AM_CFLAGS)
AM_CXXFLAGS = $(GLOBAL_AM_CXXFLAGS)
AM_FFLAGS = $(GLOBAL_AM_FFLAGS)
AM_FCFLAGS = $(GLOBAL_AM_FCFLAGS)

if STARPU_USE_CUDA
V_nvcc_  = $(V_nvcc_$(AM_DEFAULT_VERBOSITY))
V_nvcc_0 = @echo "  NVCC    " $@;
V_nvcc_1 =
V_nvcc   = $(V_nvcc_$(V))

if STARPU_COVERITY
# Avoid using nvcc when making a coverity build, nvcc produces millions of
# lines of code which we don't want to analyze.  Instead, build dumb .o files
# containing empty functions.
V_mynvcc_ = $(V_mynvcc_$(AM_DEFAULT_VERBOSITY))
V_mynvcc_0 = @echo "  myNVCC  " $@;
V_mynvcc_1 =
V_mynvcc = $(V_mynvcc_$(V))
.cu.o:
	@$(MKDIR_P) `dirname $@`
	$(V_mynvcc)grep 'extern *"C" *void *' $< | sed -ne 's/extern *"C" *void *\([a-zA-Z0-9_]*\) *(.*/void \1(void) {}/p' | $(CC) -x c - -o $@ -c
else
NVCCFLAGS += --compiler-options -fno-strict-aliasing -I$(top_builddir)/include -I$(top_srcdir)/include/ -I$(top_builddir)/src -I$(top_srcdir)/src/ $(STARPU_NVCC_H_CPPFLAGS)

.cu.cubin:
	$(V_nvcc) $(NVCC) -cubin $< -o $@ $(NVCCFLAGS)

.cu.o:
	$(V_nvcc) $(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif
endif

if STARPU_USE_HIP
V_hipcc_  = $(V_hipcc_$(AM_DEFAULT_VERBOSITY))
V_hipcc_0 = @echo "  HIPCC   " $@;
V_hipcc_1 =
V_hipcc   = $(V_hipcc_$(V))

HIPCCFLAGS += -I$(top_builddir)/include -I$(top_srcdir)/include/ -I$(top_builddir)/src -I$(top_srcdir)/src/
.hip.o:
	$(V_hipcc) $(HIPCC) $< -c -o $@ $(HIPCCFLAGS)
endif

V_icc_  = $(V_icc_$(AM_DEFAULT_VERBOSITY))
V_icc_0 = @echo "  ICC     " $@;
V_icc_1 =
V_icc   = $(V_icc_$(V))

V_ln_  = $(V_ln_$(AM_DEFAULT_VERBOSITY))
V_ln_0 = @echo "  LN      " $@;
V_ln_1 =
V_ln   = $(V_ln_$(V))

V_help2man_  = $(V_help2man_$(AM_DEFAULT_VERBOSITY))
V_help2man_0 = @echo "  HELP2MAN" $@;
V_help2man_1 =
V_help2man   = $(V_help2man_$(V))
