CFLAGS+= -g -Wall

CFLAGS+= -I../../

all: csr_interface.o blas_filters.o blas_interface.o vector_interface.o csr_filters.o vector_filters.o

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o
