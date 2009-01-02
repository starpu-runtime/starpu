STARPU= ../../starpu.a
CFLAGS+= -I../..

all: dw_strassen

strassen_kernels.o: strassen.h strassen_kernels.c
	$(CC) $(CFLAGS) strassen_kernels.c -c -o strassen_kernels.o

strassen.o: strassen.h strassen.c
	$(CC) $(CFLAGS) strassen.c -c -o strassen.o

test_strassen.o: test_strassen.c strassen.h
	$(CC) $(CFLAGS) test_strassen.c -c -o test_strassen.o

strassen_models.o: strassen_models.c strassen_models.h
	$(CC) $(CFLAGS) strassen_models.c -c -o strassen_models.o

dw_strassen: strassen_models.o strassen.o strassen_kernels.o test_strassen.o
	$(CC) strassen_models.o strassen.o strassen_kernels.o test_strassen.o -o dw_strassen $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f dw_strassen
