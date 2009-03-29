STARPU= ../../libstarpu.a
CFLAGS+= -I../../src/

CFLAGS+= -I../../

LDFLAGS+= ../common/blas.o

all: test_cholesky dw_cholesky_no_stride

dw_cholesky_models.o: dw_cholesky_models.h dw_cholesky_models.c
	$(CC) $(CFLAGS) dw_cholesky_models.c -c -o dw_cholesky_models.o

dw_cholesky_kernels.o: dw_cholesky_kernels.c
	$(CC) $(CFLAGS) dw_cholesky_kernels.c -c -o dw_cholesky_kernels.o

dw_cholesky_no_stride.o: dw_cholesky_no_stride.c
	$(CC) $(CFLAGS) dw_cholesky_no_stride.c -c -o dw_cholesky_no_stride.o

dw_cholesky.o: dw_cholesky.c
	$(CC) $(CFLAGS) dw_cholesky.c -c -o dw_cholesky.o

dw_cholesky_no_stride:  $(STARPU) dw_cholesky_no_stride.o
	$(CC) dw_cholesky_no_stride.o dw_cholesky_kernels.o dw_cholesky_models.o -o dw_cholesky_no_stride $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

test_cholesky.o: test_cholesky.c
	$(CC) $(CFLAGS) test_cholesky.c -c -o test_cholesky.o

test_cholesky: $(STARPU) test_cholesky.o dw_cholesky.o dw_cholesky_kernels.o dw_cholesky_models.o
	$(CC)  test_cholesky.o dw_cholesky.o dw_cholesky_kernels.o dw_cholesky_models.o -o dw_cholesky $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d gmon.out *.gcno *.gcda
	@rm -f dw_cholesky dw_cholesky_no_stride
