CFLAGS+= -g -Wall

CFLAGS+= -I../ -I../../

all: csr_interface.o blas_filters.o blas_interface.o vector_interface.o csr_filters.o vector_filters.o

vector_filters.o: vector_filters.c vector_filters.h
	$(CC) vector_filters.c $(CFLAGS) -c -o vector_filters.o

vector_interface.o: vector_interface.c vector_interface.h
	$(CC) vector_interface.c $(CFLAGS) -c -o vector_interface.o

blas_filters.o: blas_filters.c blas_filters.h
	$(CC) blas_filters.c $(CFLAGS) -c -o blas_filters.o

csr_interface.o: csr_interface.c csr_interface.h
	$(CC) csr_interface.c $(CFLAGS) -c -o csr_interface.o

csr_filters.o: csr_filters.c csr_filters.h
	$(CC) csr_filters.c $(CFLAGS) -c -o csr_filters.o

blas_interface.o: blas_interface.c blas_interface.h
	$(CC) blas_interface.c $(CFLAGS) -c -o blas_interface.o

clean:
	@rm -f *.o
