CFLAGS += -I../../

all: eager-central-policy.o policy.o 

eager-central-policy.o: eager-central-policy.c
	$(CC) $(CFLAGS) eager-central-policy.c -c -o eager-central-policy.o

policy.o: policy.c sched_policy.h
	$(CC) $(CFLAGS) policy.c -c -o policy.o

clean:
	@rm -f *.o
