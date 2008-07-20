CFLAGS += -I../../

all: queues.o work_stealing_queues.o central_queues.o

queues.o: queues.c queues.h
	$(CC) $(CFLAGS) queues.c -c -o queues.o

work_stealing_queues.o: work_stealing_queues.c work_stealing_queues.h
	$(CC) $(CFLAGS) work_stealing_queues.c -c -o work_stealing_queues.o

central_queues.o: central_queues.c central_queues.h
	$(CC) $(CFLAGS) central_queues.c -c -o central_queues.o

clean:
	@rm -f *.o
