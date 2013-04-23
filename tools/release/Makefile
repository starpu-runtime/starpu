# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2011  Université de Bordeaux 1
# Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
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

STARPU		?=	undefined
EXAMPLE		?= 	undefined

TARGETS	=	hello_world
TARGETS +=	block
TARGETS	+=	mult
TARGETS	+=	variable
TARGETS	+=	incrementer

ifeq ($(STARPU),undefined)
all:
	@echo
	@echo "ERROR. You need to set the variable STARPU to the name of the pkg-config StarPU package"
	@echo
clean:; rm -f $(TARGETS) *.o

else
ifeq ($(EXAMPLE),undefined)
all:
	@echo
	@echo "ERROR. You need to set the variable EXAMPLE to the directory hosting the example sources"
	@echo
clean:; rm -f $(TARGETS) *.o

else

CFLAGS          +=      $$(pkg-config --cflags $(STARPU))
LDFLAGS         +=      $$(pkg-config --libs $(STARPU))

HAS_CUDA	=	$(shell pkg-config --libs $(STARPU) |grep -i cuda)
NVCC		?=	nvcc
HAS_OPENCL	=	$(shell pkg-config --libs $(STARPU) |grep -i opencl)

%.o: $(EXAMPLE)/basic_examples/%.cu
	$(NVCC) $(CFLAGS) $< -c
%.o: $(EXAMPLE)/basic_examples/%.c
	$(CC) $(CFLAGS) $< -c

%.o: $(EXAMPLE)/incrementer/%.cu
	$(NVCC) $(CFLAGS) $< -c
%.o: $(EXAMPLE)/incrementer/%.c
	$(CC) $(CFLAGS) $< -c

all: $(TARGETS)

BLOCK_PREREQUISITES	=	block.o block_cpu.o
ifneq ($(strip $(HAS_CUDA)),)
BLOCK_PREREQUISITES	+=	block_cuda.o
BLOCK_COMPILER		=	$(NVCC)
else
BLOCK_COMPILER		=	$(CC)
endif
ifneq ($(strip $(HAS_OPENCL)),)
BLOCK_PREREQUISITES    +=	block_opencl.o
endif
block: $(BLOCK_PREREQUISITES)
	$(BLOCK_COMPILER) $(LDFLAGS) $^ -o $@

VARIABLE_PREREQUISITES	=	variable.o variable_kernels_cpu.o
ifneq ($(strip $(HAS_CUDA)),)
VARIABLE_PREREQUISITES	+=	variable_kernels.o
VARIABLE_COMPILER	=	$(NVCC)
else
VARIABLE_COMPILER	=	$(CC)
endif
ifneq ($(strip $(HAS_OPENCL)),)
VARIABLE_PREREQUISITES    +=	variable_kernels_opencl.o
endif
variable: $(VARIABLE_PREREQUISITES)
	$(VARIABLE_COMPILER) $(LDFLAGS) $^ -o $@

INCREMENTER_PREREQUISITES	=	incrementer.o
ifneq ($(strip $(HAS_CUDA)),)
INCREMENTER_PREREQUISITES	+=	incrementer_kernels.o
INCREMENTER_COMPILER		=	$(NVCC)
else
INCREMENTER_COMPILER		=	$(CC)
endif
ifneq ($(strip $(HAS_OPENCL)),)
INCREMENTER_PREREQUISITES    +=	incrementer_kernels_opencl.o
endif
incrementer: $(INCREMENTER_PREREQUISITES)
	$(INCREMENTER_COMPILER) $(LDFLAGS) $^ -o $@

clean:; rm -f $(TARGETS) *.o


endif
endif
