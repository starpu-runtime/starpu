export 

CC=gcc -m64
NVCC = nvcc -m64
CFLAGS += -g -O3 -Wall #-fPIC
CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -Wstrict-prototypes -Wnested-externs   -fno-strict-aliasing


# to use CUDA
export CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
export CUDAINSTALLDIR=/usr/local/cuda/

ifndef ATLASDIR
export ATLASDIR=/home/gonnet/DEA/BLAS/ATLAS/ATLAS/
endif
ifndef BLASARCH
export BLASARCH=Linux_UNKNOWNSSE2
endif

ifdef PERF_DEBUG
	CFLAGS += -DPERF_DEBUG
	LDFLAGS += -pg
endif

LDFLAGS+= -lm

DEPENDENCIES=comp.o threads.o mult.o jobs.o driver_core.o
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

ifdef USE_OVERLOAD
	CFLAGS += -DUSE_OVERLOAD
endif

ifeq ($(CUDA), 1)
	CFLAGS += -DUSE_CUDA 
	NVCCFLAGS +=  -DUSE_CUDA -O3 
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	NVCCFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += driver_cuda.o
	EXTRADEP += comp_cuda.cubin 
	LDFLAGS += -lcuda -L/usr/local/cuda/lib 
	LDFLAGS += -lcublas
endif

ifeq ($(CUBLAS), 1)
	CFLAGS += -DUSE_CUBLAS
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
ifdef CUBLASEMU
	LDFLAGS += -lcublasemu
else
	LDFLAGS += -lcublas
endif

	LDFLAGS += -L$(CUDASDKDIR)/lib
	LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda

	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += driver_cublas.o
endif

ifdef ATLAS
	CFLAGS+= -I$(ATLASDIR)/include/
	LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
	LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a
endif


ifdef CPUS
	CFLAGS += -DUSE_CPUS #-DDEBUG
	CFLAGS += -DNMAXCORES=$(CPUS)
	NVCCFLAGS += -DUSE_CPUS
endif

ifdef SPU
	CFLAGS += -DUSE_SPU #-DDEBUG
	CFLAGS += -DMAXSPUS=$(SPU)
	LDFLAGS += -lspe2
endif

ifdef GORDON
	CFLAGS += -DUSE_GORDON
	LDFLAGS += -lspe2
endif

ifndef FXTDIR
FXTDIR=/home/gonnet/FxT/target
endif

ifdef USE_FXT
	CFLAGS += -DUSE_FXT -I$(FXTDIR)/include/ -DCONFIG_FUT
	LDFLAGS += -lfxt -L$(FXTDIR)/lib/
endif

ifdef NO_PRIO
	CFLAGS += -DNO_PRIO
endif

all:
	@make -C common
	@make -C core
	@make -C drivers
	@make -C datawizard
	@make -C examples
	@make -C tools

clean:
	@make -C common clean
	@make -C core clean
	@make -C drivers clean
	@make -C datawizard clean
	@make -C examples clean
	@make -C tools clean

help:
	@echo "Possible options (default value) "
	@echo "  CUDA use of cuda or not (0)"
	@echo "  CUBLAS use of cublas or not (0)"
	@echo "  CPUS number of CPUs in use (0)"
	@echo "  CPUS run ATLAS functions (0)"
	@echo "  CHECK makes sure the output is correct (0)"
