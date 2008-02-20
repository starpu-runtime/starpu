CC=gcc
NVCC = nvcc
CFLAGS += -g -O3 -Wall
LDFLAGS += -lpthread


# to use CUDA
CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/
CFLAGS += -DUSE_CUDA 
# CFLAGS += -DUSE_CPUS #-DDEBUG
CFLAGS += -I$(CUDAINSTALLDIR)/include 

# for CUTILS
CUTILDIR=$(PWD)/cutil/
CUTILSRCDIR=$(CUTILDIR)/src/

# CFLAGS += -I$(CUTILDIR)/inc/

CXXFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs
CXXFLAGS +=  -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -fno-strict-aliasing 
CXXFLAGS += -I. -I$(CUDAINSTALLDIR)/include -I$(CUDASDKDIR)/inc -I$(CUTILDIR)/inc -DUNIX -O3 -g

# emu ?
# CFLAGS    += -D__DEVICE_EMULATION__
# CFLAGS      += -D__DEVICE_EMULATION__
# NVCCFLAGS   += --device-emulation   -deviceemu -DUSE_CUDA -D__DEVICE_EMULATION__ -DUNIX -DO3

NVCCFLAGS +=  -DUSE_CUDA 
#NVCCFLAGS += -DUSE_CPUS
NVCCFLAGS +=  -DUNIX -DO3 #-DO3
NVCCFLAGS += -I$(CUTILDIR)/inc/

LDFLAGS += -L$(CUDASDKDIR)/lib
LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda
LDFLAGS += -lGL -lGLU


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

mult: comp.o mult.o mult_cuda.o jobs.o comp_cuda.cubin mult.h util.h parameters.h
	$(CC) $(CFLAGS) $(LDFLAGS) mult.o mult_cuda.o comp.o jobs.o -o mult

clean:
	rm -f *.o *.cu_o *.cubin *.cpp_o
	rm -f mult
