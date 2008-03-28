CFLAGS+= -g -Wall


all: test

coherency.o: coherency.c coherency.h

copy-driver.o: copy-driver.c

test.o: test.c

test: test.o coherency.o copy-driver.o

clean:
	rm -f *.o test
