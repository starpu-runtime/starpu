CC=gcc
NVCC = nvcc
CFLAGS += -g -O3 -Wall #-fPIC
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
	OBJDEPS += ../drivers/cuda/mult_cuda.o
	DEPENDENCIES += mult_cuda.o
	EXTRADEP += comp_cuda.cubin 
endif

ifeq ($(CUBLAS), 1)
	CFLAGS += -DUSE_CUBLAS
	LDFLAGS += -lcublas
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += mult_cublas.o
	OBJDEPS += ../drivers/cublas/mult_cublas.o
endif

ifdef ATLAS
	CFLAGS += -DUSE_CPU_BLAS
endif


ifdef CPUS
	CFLAGS += -DUSE_CPUS #-DDEBUG
	CFLAGS += -DNMAXCORES=$(CPUS)
	NVCCFLAGS += -DUSE_CPUS
	OBJDEPS += ../drivers/core/mult_core.o
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

LDFLAGS += -L../core/ -L../common/
CFLAGS += -I../

OBJDEPS += ../common/util.o ../common/threads.o ../core/jobs.o ../core/workers.o

ATLASDIR=/home/gonnet/DEA/BLAS/ATLAS/ATLAS/
BLASARCH=Linux_UNKNOWNSSE2

CFLAGS+= -I$(ATLASDIR)/include/
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a



all: multiplication multiplication2 factoLU

multiplication2.o:
	$(CC) $(CFLAGS) multiplication2.c -c -o multiplication2.o

multiplication2: multiplication2.o
	$(CC) multiplication2.o -o multiplication2 $(OBJDEPS) $(LDFLAGS) 


multiplication.o:
	$(CC) $(CFLAGS) multiplication.c -c -o multiplication.o

multiplication: multiplication.o
	$(CC) multiplication.o -o multiplication $(OBJDEPS) $(LDFLAGS) 

factoLU: factoLU.o
	$(CC) factoLU.o -o factoLU $(OBJDEPS) $(LDFLAGS) 

factoLU.o: factoLU.c
	$(CC) $(CFLAGS) factoLU.c -c -o factoLU.o

clean:
	rm -f multiplication multiplication2 factoLU *.o
