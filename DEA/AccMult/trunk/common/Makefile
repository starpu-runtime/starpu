CC=gcc
NVCC = nvcc
CFLAGS += -g -O3 -Wall #-fPIC
#CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -Wmain  -fno-strict-aliasing

CFLAGS += -I../ -I../core/

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

ifeq ($(CUDA), 1)
	CFLAGS += -DUSE_CUDA 
	NVCCFLAGS +=  -DUSE_CUDA 
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	NVCCFLAGS += -I$(CUDAINSTALLDIR)/include 
endif

ifeq ($(CUBLAS), 1)
	CFLAGS += -DUSE_CUBLAS
ifdef CUBLASEMU
	LDFLAGS += -lcublasemu
else
	LDFLAGS += -lcublas
endif
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
endif

all: matrix.o threads.o

threads.o: threads.c threads.h
	$(CC) $(CFLAGS) threads.c -c -o threads.o

matrix.o: matrix.c util.h
	$(CC) $(CFLAGS) matrix.c -c -o matrix.o

clean:
	@rm -f *.o 

