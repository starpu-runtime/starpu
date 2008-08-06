RIBSCHED= ../../ribsched.a
CFLAGS+= -I../..

all: test_cholesky

dw_cholesky_kernels.o: dw_cholesky_kernels.c
	$(CC) $(CFLAGS) dw_cholesky_kernels.c -c -o dw_cholesky_kernels.o

dw_cholesky.o: dw_cholesky.c
	$(CC) $(CFLAGS) dw_cholesky.c -c -o dw_cholesky.o

test_cholesky.o: test_cholesky.c
	$(CC) $(CFLAGS) test_cholesky.c -c -o test_cholesky.o

test_cholesky: test_cholesky.o dw_cholesky.o dw_cholesky_kernels.o
	$(CC)  test_cholesky.o dw_cholesky.o dw_cholesky_kernels.o -o dw_cholesky $(LDFLAGS) $(RIBSCHED)

clean:
	@rm -f *.o *.d
	@rm -f dw_cholesky
