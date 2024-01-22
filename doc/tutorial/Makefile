# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
CFLAGS          +=      $$(pkg-config --cflags starpu-1.4)
LDLIBS          +=      $$(pkg-config --libs starpu-1.4)

HAS_CUDA	=	$(shell starpu_config | grep 'STARPU_USE_CUDA 1')
NVCC		?=	nvcc -std=c++11
HAS_OPENCL	=	$(shell starpu_config | grep 'STARPU_USE_OPENCL 1')

%.o: %.cu
	$(NVCC) $(CFLAGS) $< -c

TARGETS = hello_world vector_scal

all: $(TARGETS)

VECTOR_SCAL_PREREQUISITES	=	vector_scal.o vector_scal_cpu.o
ifneq ($(strip $(HAS_CUDA)),)
VECTOR_SCAL_PREREQUISITES	+=	vector_scal_cuda.o
VECTOR_SCAL_COMPILER		=	$(NVCC)
else
VECTOR_SCAL_COMPILER		=	$(CC)
endif
ifneq ($(strip $(HAS_OPENCL)),)
VECTOR_SCAL_PREREQUISITES 	+=	vector_scal_opencl.o
LDLIBS				+=	-lOpenCL
endif

vector_scal: $(VECTOR_SCAL_PREREQUISITES)
	$(VECTOR_SCAL_COMPILER) $^ $(LDLIBS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o
