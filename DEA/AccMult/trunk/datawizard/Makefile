.PHONY: spu

CFLAGS+= -g -Wall

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

memalloc.o: memalloc.c memalloc.h
	$(CC) memalloc.c $(CFLAGS) -c -o memalloc.o

hierarchy.o: hierarchy.c hierarchy.h coherency.h
	$(CC) hierarchy.c $(CFLAGS) -c -o hierarchy.o

coherency.o: coherency.c coherency.h coherency_common.c
	$(CC) coherency.c $(CFLAGS) -c -o coherency.o

copy-driver.o: copy-driver.c
	$(CC) copy-driver.c $(CFLAGS) -c -o copy-driver.o

test.o: test.c

test: test.o coherency.o copy-driver.o hierarchy.o filters.o
	$(CC) test.o coherency.o copy-driver.o  hierarchy.o filters.o -lpthread -o test

clean:
	@rm -f *.o test
	@make -C spu clean
