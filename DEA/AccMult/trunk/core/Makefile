CFLAGS += -I../

CFLAGS += -I../common/

LDFLAGS += -L../datawizard/
CFLAGS += -I../datawizard/

all: jobs.o workers.o
	@make -C dependencies
	@make -C mechanisms
	@make -C policies

jobs.o: jobs.c
	$(CC) $(CFLAGS) jobs.c -c -o jobs.o

workers.o: workers.c
	$(CC) $(CFLAGS) workers.c -c -o workers.o

clean:
	@make -C policies clean
	@make -C mechanisms clean
	@make -C dependencies clean
	@rm -f *.o
