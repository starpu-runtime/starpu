CFLAGS += -I../

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@
	$(CC) $(CFLAGS) $< -MM > $*.d

all: jobs.o workers.o
	@make -C dependencies
	@make -C mechanisms
	@make -C policies

clean:
	@make -C policies clean
	@make -C mechanisms clean
	@make -C dependencies clean
	@rm -f *.o *.d
