CC=gcc
NVCC = nvcc

CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/
NVCCFLAGS +=  -DUNIX -DO3

CFLAGS += -g -O3 -Wall
CFLAGS += -I$(CUDAINSTALLDIR)/include  


LDFLAGS += -L$(CUDASDKDIR)/lib
LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda

all: bandwith-test

kernel_code.cubin: kernel_code.cu
	$(NVCC) -cubin kernel_code.cu -o kernel_code.cubin -m32 --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

host_code.o: host_code.c 
	$(NVCC) $(NVCCFLAGS)  $(LDFLAGS) host_code.c -c -o host_code.o

bandwith-test: host_code.o kernel_code.cubin
	$(CC) $(LDFLAGS) host_code.o -o bandwith-test

clean:
	rm -f *.o *.cu_o *.cubin
	rm -f bandwith-test
