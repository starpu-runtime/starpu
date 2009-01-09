CFLAGS+= -Iexternals/scalp/
CFLAGS+= -I../..

all: driver_gordon.o

driver_gordon.o: driver_gordon.c driver_gordon.h
	$(CC) $(CFLAGS) driver_gordon.c -c -o driver_gordon.o

clean:   
	@rm -f *.o
