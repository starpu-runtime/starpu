CFLAGS+= -g -Wall

CFLAGS+= -I../

ifeq ($(CUBLAS), 1)
	CFLAGS+= -I$(CUDAINSTALLDIR)/include
	CFLAGS+= -DUSE_CUBLAS
endif


all: coherency.o copy-driver.o 
	@make -C spu

coherency.o: coherency.c coherency.h
	$(CC) coherency.c $(CFLAGS) -c -o coherency.o

copy-driver.o: copy-driver.c
	$(CC) copy-driver.c $(CFLAGS) -c -o copy-driver.o

test.o: test.c

test: test.o coherency.o copy-driver.o
	$(CC) test.o coherency.o copy-driver.o -lpthread -o test

clean:
	@rm -f *.o test
	@make -C spu clean
