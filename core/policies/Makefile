CFLAGS += -I../../

OBJS := no-prio-policy.o eager-central-policy.o eager-central-priority-policy.o work-stealing-policy.o sched_policy.o deque-modeling-policy.o random-policy.o deque-modeling-policy-data-aware.o 

all: $(OBJS)

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d *.gcno *.gcda
