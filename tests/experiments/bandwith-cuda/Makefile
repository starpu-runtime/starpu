CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/

CFLAGS += -I$(CUDAINSTALLDIR)/include -Wall
LDFLAGS += -lcuda -L/usr/local/cuda/lib
LDFLAGS += -lcublas -lpthread

CFLAGS+= -I$(ATLASDIR)/include/
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a

CFLAGS+= -g #-pg
#LDFLAGS+= -pg

all:  cuda-bandwith

cuda-bandwith:  cuda-bandwith.o
	$(CC)  cuda-bandwith.o -o  cuda-bandwith $(LDFLAGS)

clean:
	rm -f cuda-bandwith
	rm -f *.o
