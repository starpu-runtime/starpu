CFLAGS += -I../../

OBJS := eager-central-policy.o eager-central-priority-policy.o work-stealing-policy.o sched_policy.o deque-modeling-policy.o random-policy.o

all: $(OBJS)

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d
