STARPU= ../../libstarpu.a

LDFLAGS += ../common/blas.o

all: strassen

strassen.o: strassen.c
	$(CC) $(CFLAGS) strassen.c -c -o strassen.o

strassen_kernels.o: strassen_kernels.c
	$(CC) $(CFLAGS) strassen_kernels.c -c -o strassen_kernels.o

strassen: $(STARPU) strassen.o strassen_kernels.o
	$(CC) strassen.o strassen_kernels.o -o strassen $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f strassen
