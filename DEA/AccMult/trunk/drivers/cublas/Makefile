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


DEPENDENCIES=comp.o threads.o mult.o jobs.o mult_core.o matrix.o
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

NVCCFLAGS +=  -DUNIX -DO3
NVCCFLAGS += -I$(CUTILDIR)/inc/


CFLAGS += -I../../

all: mult_cublas.o

mult_cublas.o: mult_cublas.c mult_cublas.h
	$(CC) $(CFLAGS) mult_cublas.c -c -o mult_cublas.o

clean:
	@rm -f *.o *.cu_o *.cubin
