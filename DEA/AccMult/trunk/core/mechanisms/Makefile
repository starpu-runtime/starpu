CFLAGS += -I../../

all: queues.o work_stealing_queues.o central_queues.o central_queues_priorities.o

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o
