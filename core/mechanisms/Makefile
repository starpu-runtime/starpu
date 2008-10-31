CFLAGS += -I../../

all: queues.o deque_queues.o priority_queues.o fifo_queues.o

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o *.gcno *.gcda
