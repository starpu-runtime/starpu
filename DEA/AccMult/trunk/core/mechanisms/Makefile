CFLAGS += -I../../

all: queues.o

queues.o: queues.c queues.h
	$(CC) $(CFLAGS) queues.c -c -o queues.o

clean:
	@rm -f *.o
