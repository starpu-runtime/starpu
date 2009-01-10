STARPU= ../../starpu.a
CFLAGS+= -I../..
 
ifeq ($(CUDA), 1)
        EXTRADEP += ../cuda/incrementer_cuda.cubin
endif

all: $(EXTRADEP) incrementer

../cuda/incrementer_cuda.cubin:
	make -C ../cuda incrementer_cuda.cubin

incrementer.o: incrementer.c
	$(CC) $(CFLAGS) incrementer.c -c -o incrementer.o

incrementer: $(STARPU) incrementer.o
	$(CC) incrementer.o -o incrementer $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f incrementer
