# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
