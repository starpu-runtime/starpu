CC=gcc
NVCC = nvcc
CFLAGS += -g -O3 -Wall #-fPIC
#CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -Wmain  -fno-strict-aliasing

# to use CUDA
CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/

LDFLAGS+= -lm

ifdef DONTBIND
	CFLAGS += -DDONTBIND
endif

ifeq ($(CUDA), 1)
	CFLAGS += -DUSE_CUDA 
	NVCCFLAGS +=  -DUSE_CUDA 
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	NVCCFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += mult_cuda.o
	EXTRADEP += comp_cuda.cubin 
endif

NVCCFLAGS +=  -DUNIX -DO3
NVCCFLAGS += -I$(CUTILDIR)/inc/

LDFLAGS += -L$(CUDASDKDIR)/lib
LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda

CFLAGS += -I../../
NVCCFLAGS += -I../../

all: mult_cuda.o comp_cuda.cubin

mult_cuda.o: mult_cuda.c mult_cuda.h
	$(NVCC) $(NVCCFLAGS) mult_cuda.c -c -o mult_cuda.o

comp_cuda.cubin: comp_cuda.cu comp_cuda.h
	$(NVCC) -cubin comp_cuda.cu -o comp_cuda.cubin -m32 --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

clean:
	rm -f *.o *.cu_o *.cubin
