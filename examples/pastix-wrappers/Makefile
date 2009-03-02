STARPU= ../../starpu.a
CFLAGS+= -I../.. -g
CFLAGS+= -I../../src/

all: starpu-blas-wrapper.a

starpu-blas-wrapper.o: starpu-blas-wrapper.c starpu-blas-wrapper.h
	$(CC) $(CFLAGS) starpu-blas-wrapper.c -c -o starpu-blas-wrapper.o

starpu-blas-wrapper.a: starpu-blas-wrapper.o
	$(AR) rcs starpu-blas-wrapper.a starpu-blas-wrapper.o

clean:
	@rm -f *.a *.o *.d *.gcno *.gcda
