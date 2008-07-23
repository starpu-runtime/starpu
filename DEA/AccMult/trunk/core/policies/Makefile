CFLAGS += -I../../

all: eager-central-policy.o eager-central-priority-policy.o work-stealing-policy.o policy.o 

eager-central-priority-policy.o: eager-central-priority-policy.c eager-central-priority-policy.h
	$(CC) $(CFLAGS) eager-central-priority-policy.c -c -o eager-central-priority-policy.o

eager-central-policy.o: eager-central-policy.c eager-central-policy.h
	$(CC) $(CFLAGS) eager-central-policy.c -c -o eager-central-policy.o

work-stealing-policy.o: work-stealing-policy.c work-stealing-policy.h
	$(CC) $(CFLAGS) work-stealing-policy.c -c -o work-stealing-policy.o

policy.o: policy.c sched_policy.h
	$(CC) $(CFLAGS) policy.c -c -o policy.o

clean:
	@rm -f *.o
