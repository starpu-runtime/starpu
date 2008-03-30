CC=gcc
NVCC = nvcc
CFLAGS += -g -O0 -Wall #-fPIC
#CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -Wmain  -fno-strict-aliasing

# to use CUDA
CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/

# for compile time tuning 
DEFAULTSIZE=2048
DEFAULTGRAIN=128

DEFAULTNSAMPLE=10

ifdef NSAMPLE
	CFLAGS += -DNSAMPLE=$(NSAMPLE)
else
	CFLAGS += -DNSAMPLE=$(DEFAULTNSAMPLE)
endif

ATLASDIR=/home/gonnet/DEA/BLAS/ATLAS/ATLAS/
BLASARCH=Linux_UNKNOWNSSE2

CFLAGS+= -I$(ATLASDIR)/include/
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a

LDFLAGS+= -lm


DEPENDENCIES=comp.o threads.o mult.o jobs.o mult_core.o util.o
EXTRADEP=

ifeq ($(MARCEL), 1)
	CC=$(shell pm2-config --cc)
	LDFLAGS += $(shell pm2-config --libs) 
	CFLAGS += -DUSE_MARCEL $(shell pm2-config --cflags)
else
	LDFLAGS += -lpthread
endif

ifdef DONTBIND
	CFLAGS += -DDONTBIND
endif

ifeq ($(CHECK), 1)
	CFLAGS += -DCHECK_OUTPUT
endif

ifdef SIZE 
	CFLAGS += -DSIZE=$(SIZE)
else
	CFLAGS += -DSIZE=$(DEFAULTSIZE)
endif

ifeq ($(CUDA), 1)
	CFLAGS += -DUSE_CUDA 
	NVCCFLAGS +=  -DUSE_CUDA 
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	NVCCFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += mult_cuda.o
	EXTRADEP += comp_cuda.cubin 
endif

ifeq ($(CUBLAS), 1)
	CFLAGS += -DUSE_CUBLAS
ifdef CUBLASEMU
	LDFLAGS += -lcublasemu
else
	LDFLAGS += -lcublas
endif
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += mult_cublas.o
endif

ifdef ATLAS
	CFLAGS += -DUSE_CPU_BLAS
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

#NVCCFLAGS +=  -DUNIX -DO3
#NVCCFLAGS += -I$(CUTILDIR)/inc/
#
#LDFLAGS += -L$(CUDASDKDIR)/lib
#LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda


all:
	make -C common
	make -C core
	make -C drivers
	make -C datawizard
	make -C examples

clean:
	make -C common clean
	make -C core clean
	make -C drivers clean
	make -C datawizard clean
	make -C examples clean


help:
	@echo "Possible options (default value) "
	@echo "  SIZE size of the problem ($(DEFAULTSIZE))"
	@echo "  CUDA use of cuda or not (0)"
	@echo "  CUBLAS use of cublas or not (0)"
	@echo "  CPUS number of CPUs in use (0)"
	@echo "  CPUS run ATLAS functions (0)"
	@echo "  CHECK makes sure the output is correct (0)"
	@echo "  GRAIN job grain size ($(DEFAULTGRAIN))"
