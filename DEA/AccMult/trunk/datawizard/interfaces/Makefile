CFLAGS+= -g -Wall

CFLAGS+= -I../ -I../../

ifeq ($(CUBLAS), 1)
	CFLAGS+= -I$(CUDAINSTALLDIR)/include
	CFLAGS+= -DUSE_CUBLAS
endif

all: blas_filters.o blas_interface.o

blas_filters.o: blas_filters.c blas_filters.h
	$(CC) blas_filters.c $(CFLAGS) -c -o blas_filters.o

blas_interface.o: blas_interface.c blas_interface.h
	$(CC) blas_interface.c $(CFLAGS) -c -o blas_interface.o

clean:
	@rm -f *.o
