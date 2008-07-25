.PHONY: interfaces spu

CFLAGS+= -I../

ifeq ($(CUBLAS), 1)
	CFLAGS+= -I$(CUDAINSTALLDIR)/include
	CFLAGS+= -DUSE_CUBLAS
endif

ifdef SPU
	SPU_TARGET=spu
endif

all: interfaces coherency.o copy-driver.o hierarchy.o memalloc.o $(SPU_TARGET)

interfaces:
	@make -C interfaces

spu:
	@make -C spu

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o test
	@make -C interfaces clean
	@make -C spu clean
