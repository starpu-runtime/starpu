CFLAGS+= -I$(ATLASDIR)/include/
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a

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

ifeq ($(CUBLAS), 1)
	CFLAGS += -DUSE_CUBLAS
ifdef CUBLASEMU
	LDFLAGS += -lcublasemu
else	
	LDFLAGS += -lcublas
endif
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += driver_cublas.o
endif

NVCCFLAGS +=  -DUNIX -DO3
NVCCFLAGS += -I$(CUTILDIR)/inc/


CFLAGS += -I../../

all: driver_cublas.o

driver_cublas.o: driver_cublas.c driver_cublas.h
	$(CC) $(CFLAGS) driver_cublas.c -c -o driver_cublas.o

clean:
	@rm -f *.o *.cu_o *.cubin
