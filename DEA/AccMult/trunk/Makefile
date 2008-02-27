CC=gcc
NVCC = nvcc
CFLAGS += -g -O3 -Wall -fPIC
CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -Wmain  -fno-strict-aliasing

CFLAGS+= -DUSE_CPU_BLAS
CFLAGS+= -DUSE_CUBLAS

ATLASDIR=/home/gonnet/DEA/BLAS/ATLAS/ATLAS/
BLASARCH=Linux_UNKNOWNSSE2

CFLAGS+= -I$(ATLASDIR)/include/
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a

LDFLAGS+= -lm


ifeq ($(MARCEL), 1)
	CC=$(shell pm2-config --cc)
	LDFLAGS += $(shell pm2-config --libs) 
	CFLAGS += -DUSE_MARCEL $(shell pm2-config --cflags)
else
	LDFLAGS += -lpthread
endif

# to use CUDA
CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/


# for compile time tuning 

DEFAULTSIZE=2048
DEFAULTGRAIN=128

ifdef SIZE 
	CFLAGS += -DSIZE=$(SIZE)
else
	CFLAGS += -DSIZE=$(DEFAULTSIZE)
endif

ifneq ($(CUDA), 0)
	CFLAGS += -DUSE_CUDA 
	NVCCFLAGS +=  -DUSE_CUDA 
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	NVCCFLAGS += -I$(CUDAINSTALLDIR)/include 
endif

ifdef CPUS
	CFLAGS += -DUSE_CPUS #-DDEBUG
	CFLAGS += -DNMAXCORES=$(CPUS)
	NVCCFLAGS += -DUSE_CPUS
endif

ifdef GRAIN
	CFLAGS += -DGRAIN=$(GRAIN)
	NVCCFLAGS += -DGRAIN=$(GRAIN)
else
	CFLAGS += -DGRAIN=$(DEFAULTGRAIN)
	NVCCFLAGS += -DGRAIN=$(DEFAULTGRAIN)
endif

NVCCFLAGS +=  -DUNIX -DO3
NVCCFLAGS += -I$(CUTILDIR)/inc/

LDFLAGS += -L$(CUDASDKDIR)/lib
LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda

LDFLAGS += -lcublas

all: mult

mult_cuda.o: mult_cuda.c mult_cuda.h mult.h  parameters.h
	$(NVCC) $(NVCCFLAGS) mult_cuda.c -c -o mult_cuda.o

comp_cuda.cubin: comp_cuda.cu comp_cuda.h mult.h  parameters.h
	$(NVCC) -cubin comp_cuda.cu -o comp_cuda.cubin -m32 --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

mult.o: mult.c mult.h  parameters.h
	$(CC) $(CFLAGS) mult.c -c -o mult.o

comp.o: comp.c comp.h mult.h  parameters.h
	$(CC) $(CFLAGS) comp.c -c -o comp.o

jobs.o: jobs.c jobs.h mult.h parameters.h
	$(CC) $(CFLAGS) jobs.c -c -o jobs.o

mult: comp.o mult.o mult_cuda.o jobs.o mult.h util.h parameters.h
	$(CC) $(CFLAGS) mult.o mult_cuda.o comp.o jobs.o -o mult $(LDFLAGS) 

clean:
	rm -f *.o *.cu_o *.cubin *.cpp_o
	rm -f mult

help:
	@echo "Possible options (default value) "
	@echo "  SIZE size of the problem ($(DEFAULTSIZE))"
	@echo "  CUDA use of cuda or not (1)"
	@echo "  CPUS number of CPUs in use (0)"
	@echo "  GRAIN job grain size ($(DEFAULTGRAIN))"
