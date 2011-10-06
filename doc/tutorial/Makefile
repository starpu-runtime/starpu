# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2011  Université de Bordeaux 1
# Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
#

CFLAGS          +=      $$(pkg-config --cflags libstarpu)
LDFLAGS         +=      $$(pkg-config --libs libstarpu)

HAS_CUDA	=	$(shell pkg-config --libs libstarpu|grep -i cuda)
NVCC		?=	nvcc
HAS_OPENCL	=	$(shell pkg-config --libs libstarpu|grep -i opencl)

%.o: %.cu
	nvcc $(CFLAGS) $< -c

all: hello_world vector_scal

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
	$(VECTOR_SCAL_COMPILER) $(LDFLAGS) $^ -o $@

clean:
	rm -f hello_world vector_scal *.o

