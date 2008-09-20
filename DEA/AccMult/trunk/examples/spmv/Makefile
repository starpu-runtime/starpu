RIBSCHED= ../../ribsched.a
CFLAGS+= -I../..

ifeq ($(CUDA), 1)
	EXTRADEP += ../cuda/spmv_cuda.cubin
endif

all: $(EXTRADEP) dw_spmv example_read
	make -C matrix-market

../cuda/spmv_cuda.cubin:
	make -C ../cuda spmv_cuda.cubin

dw_spmv.o: dw_spmv.c dw_spmv.h
	$(CC) $(CFLAGS) dw_spmv.c -c -o dw_spmv.o

dw_spmv: dw_spmv.o
	$(CC) dw_spmv.o -o dw_spmv $(LDFLAGS) $(RIBSCHED)

clean:
	@make -C matrix-market clean
	@rm -f *.o *.d
	@rm -f dw_spmv
