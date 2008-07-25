CFLAGS += -I../../

all: eager-central-policy.o eager-central-priority-policy.o work-stealing-policy.o sched_policy.o 

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o
