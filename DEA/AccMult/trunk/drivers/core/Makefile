CFLAGS += -I../../

all: driver_core.o

driver_core.o: driver_core.c  driver_core.h
	$(CC) $(CFLAGS) driver_core.c -c -o driver_core.o

clean:
	@rm -f *.o 
