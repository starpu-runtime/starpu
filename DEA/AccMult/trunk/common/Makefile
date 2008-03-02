CC=gcc
NVCC = nvcc
CFLAGS += -g -O3 -Wall #-fPIC
#CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -Wmain  -fno-strict-aliasing

CFLAGS += -I../

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
	LDFLAGS += -lcublas
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
endif

all: util.o threads.o

threads.o: threads.c threads.h
	$(CC) $(CFLAGS) threads.c -c -o threads.o

util.o: util.c util.h
	$(CC) $(CFLAGS) util.c -c -o util.o

clean:
	rm -f *.o 
