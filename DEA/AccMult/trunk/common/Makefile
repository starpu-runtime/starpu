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

all: matrix.o threads.o mutex.o rwlock.o timing.o

mutex.o: mutex.c mutex.h
	$(CC) $(CFLAGS) mutex.c -c -o mutex.o

rwlock.o: rwlock.c rwlock.h
	$(CC) $(CFLAGS) rwlock.c -c -o rwlock.o

threads.o: threads.c threads.h
	$(CC) $(CFLAGS) threads.c -c -o threads.o

matrix.o: matrix.c util.h
	$(CC) $(CFLAGS) matrix.c -c -o matrix.o

timing.o: timing.c timing.h
	$(CC) $(CFLAGS) timing.c -c -o timing.o

clean:
	@rm -f *.o 
