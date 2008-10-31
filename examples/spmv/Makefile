RIBSCHED= ../../ribsched.a
CFLAGS+= -I../..

ifeq ($(CUDA), 1)
	EXTRADEP += ../cuda/spmv_cuda.cubin
endif

all: $(EXTRADEP) dw_spmv dw_block_spmv
	make -C matrix-market

../cuda/spmv_cuda.cubin:
	make -C ../cuda spmv_cuda.cubin

dw_spmv.o: dw_spmv.c dw_spmv.h
	$(CC) $(CFLAGS) dw_spmv.c -c -o dw_spmv.o

dw_spmv: dw_spmv.o
	$(CC) dw_spmv.o -o dw_spmv $(LDFLAGS) $(RIBSCHED)

dw_block_spmv: dw_block_spmv.o dw_block_spmv_kernels.o
	make -C matrix-market mm_to_bcsr.o
	make -C matrix-market mmio.o
	$(CC) dw_block_spmv.o dw_block_spmv_kernels.o matrix-market/mm_to_bcsr.o matrix-market/mmio.o -o dw_block_spmv $(LDFLAGS) $(RIBSCHED)

clean:
	@make -C matrix-market clean
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f dw_spmv dw_block_spmv
