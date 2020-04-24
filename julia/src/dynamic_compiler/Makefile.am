# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

LD=$(CC)

ifeq ($(STARPU_USE_CUDA),yes)
        LD := ${NVCC}
endif

AM_CFLAGS = -O3 -g -DSTRIDE=${STRIDE} $(shell pkg-config --cflags starpu-1.3)
AM_CFLAGS += ${CFLAGS} -Wall -mavx -fomit-frame-pointer -march=native -ffast-math
CUDA_CFLAGS = ${CFLAGS}
#LDFLAGS +=$(shell pkg-config --libs starpu-1.3)

EXTERNLIB=extern_tasks.so
GENERATEDLIB=generated_tasks.so

C_OBJECTS=$(patsubst %.c,%.o,$(wildcard gen*.c))
CUDA_OBJECTS=$(patsubst %.cu,%.o,$(wildcard gen*.cu))
ifneq ($(STARPU_USE_CUDA_CUDA),yes)
	CUDA_OBJECTS:=
endif

%.o: %.c
	$(CC) -c -fPIC $(CPU_CFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) -dc $(CUDA_CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

#${EXTERNLIB}: cpu_mult.c
#	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

${GENERATEDLIB}: $(C_OBJECTS) $(CUDA_OBJECTS)
	$(LD) -shared $(LDFLAGS) $^ -o $@

