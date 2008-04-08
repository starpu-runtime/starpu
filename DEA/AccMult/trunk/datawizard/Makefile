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

all: coherency.o copy-driver.o hierarchy.o filters.o $(SPU_TARGET)

spu:
	@make -C spu

filters.o: filters.c filters.h
	$(CC) filters.c $(CFLAGS) -c -o filters.o

hierarchy.o: hierarchy.c hierarchy.h coherency.h
	$(CC) hierarchy.c $(CFLAGS) -c -o hierarchy.o

coherency.o: coherency.c coherency.h coherency_common.c
	$(CC) coherency.c $(CFLAGS) -c -o coherency.o

copy-driver.o: copy-driver.c
	$(CC) copy-driver.c $(CFLAGS) -c -o copy-driver.o

test.o: test.c

test: test.o coherency.o copy-driver.o
	$(CC) test.o coherency.o copy-driver.o -lpthread -o test

clean:
	@rm -f *.o test
	@make -C spu clean
