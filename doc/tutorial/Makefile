# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
CFLAGS          +=      $$(pkg-config --cflags starpu-1.1)
LDLIBS          +=      $$(pkg-config --libs starpu-1.1)

HAS_CUDA	=	$(shell pkg-config --libs starpu-1.1 |grep -i cuda)
NVCC		?=	nvcc
HAS_OPENCL	=	$(shell pkg-config --libs starpu-1.1 |grep -i opencl)

%.o: %.cu
	nvcc $(CFLAGS) $< -c

TARGETS = hello_world vector_scal vector_scal_task_insert

all: $(TARGETS)

VECTOR_SCAL_PREREQUISITES	=	vector_scal.o vector_scal_cpu.o
ifneq ($(strip $(HAS_CUDA)),)
VECTOR_SCAL_PREREQUISITES	+=	vector_scal_cuda.o
VECTOR_SCAL_COMPILER		=	$(NVCC)
else
VECTOR_SCAL_COMPILER		=	$(CC)
endif
ifneq ($(strip $(HAS_OPENCL)),)
VECTOR_SCAL_PREREQUISITES += vector_scal_opencl.o
endif

vector_scal: $(VECTOR_SCAL_PREREQUISITES)
	$(VECTOR_SCAL_COMPILER) $^ $(LDLIBS) -o $@

VECTOR_SCAL_TASK_INSERT_PREREQUISITES		=	vector_scal_task_insert.o vector_scal_cpu.o
ifneq ($(strip $(HAS_CUDA)),)
VECTOR_SCAL_TASK_INSERT_PREREQUISITES		+=	vector_scal_cuda.o
VECTOR_SCAL_TASK_INSERT_COMPILER		=	$(NVCC)
else
VECTOR_SCAL_TASK_INSERT_COMPILER		=	$(CC)
endif
ifneq ($(strip $(HAS_OPENCL)),)
VECTOR_SCAL_TASK_INSERT_PREREQUISITES += vector_scal_opencl.o
endif

vector_scal_task_insert: $(VECTOR_SCAL_TASK_INSERT_PREREQUISITES)
	$(VECTOR_SCAL_TASK_INSERT_COMPILER) $^ $(LDLIBS) -o $@

clean:
	rm -f $(TARGETS) *.o
