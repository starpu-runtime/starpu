CFLAGS+= -g -Wall

CFLAGS+= -I../

all: coherency.o copy-driver.o test

coherency.o: coherency.c coherency.h

copy-driver.o: copy-driver.c

test.o: test.c

test: test.o coherency.o copy-driver.o
	$(CC) test.o coherency.o copy-driver.o -lpthread -o test

clean:
	rm -f *.o test
