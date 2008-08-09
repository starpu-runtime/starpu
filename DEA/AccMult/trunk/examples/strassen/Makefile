RIBSCHED= ../../ribsched.a
CFLAGS+= -I../..

all: dw_strassen

strassen_kernels.o: strassen.h strassen_kernels.c
	$(CC) $(CFLAGS) strassen_kernels.c -c -o strassen_kernels.o

strassen.o: strassen.h strassen.c
	$(CC) $(CFLAGS) strassen.c -c -o strassen.o

test_strassen.o: test_strassen.c strassen.h
	$(CC) $(CFLAGS) test_strassen.c -c -o test_strassen.o

dw_strassen: strassen.o strassen_kernels.o test_strassen.o
	$(CC) strassen.o strassen_kernels.o test_strassen.o -o dw_strassen $(LDFLAGS) $(RIBSCHED)

clean:
	@rm -f *.o *.d
	@rm -f dw_strassen
