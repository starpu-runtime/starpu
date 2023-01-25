# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

include $(top_srcdir)/make/starpu-notests.mk

LD=$(CC_OR_NVCC)
AM_CPPFLAGS = -I$(abs_top_srcdir)/include/ -I$(abs_top_builddir)/src -I$(abs_top_builddir)/include \
	 -I$(abs_top_srcdir)/julia/src/ $(STARPU_H_CPPFLAGS)

AM_CFLAGS += -fPIC -DSTRIDE=${STRIDE} -mavx -fomit-frame-pointer -march=native -ffast-math
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@
LIBS += -L $(top_builddir)/julia/src/libstarpujulia-$(STARPU_EFFECTIVE_VERSION).1.3
CUDA_CFLAGS = $(STARPU_CUDA_CPPFLAGS) -Wno-deprecated-gpu-targets
EXTERNLIB=extern_tasks.so
GENERATEDLIB=generated_tasks.so

C_OBJECTS=$(patsubst %.c,%.o,$(wildcard gen*.c))

if STARPU_USE_CUDA
CUDA_OBJECTS=$(patsubst %.cu,%.o,$(wildcard gen*.cu))
else
CUDA_OBJECTS=
endif

%.o: %.c
	$(CC) -c $(AM_CPPFLAGS) $(AM_CFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) -dc $(AM_CPPFLAGS) $(CUDA_CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

${EXTERNLIB}: $(SOURCES_CPU)
	$(CC) $(AM_CPPFLAGS) $(AM_CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

${GENERATEDLIB}: $(C_OBJECTS) $(CUDA_OBJECTS)
	$(LD) -shared $^ -o $@ $(LDFLAGS)

