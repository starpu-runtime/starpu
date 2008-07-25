.PHONY: interfaces spu

CFLAGS+= -I../

ifeq ($(CUBLAS), 1)
	CFLAGS+= -I$(CUDAINSTALLDIR)/include
	CFLAGS+= -DUSE_CUBLAS
endif

ifdef SPU
	SPU_TARGET=spu
endif

OBJS := coherency.o copy-driver.o hierarchy.o memalloc.o

all: interfaces $(SPE_TARGET) $(OBJS)

interfaces:
	@make -C interfaces

spu:
	@make -C spu

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d
	@make -C interfaces clean
	@make -C spu clean
