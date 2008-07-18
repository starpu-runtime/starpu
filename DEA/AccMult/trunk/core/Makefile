CFLAGS += -I../

CFLAGS += -I../common/

LDFLAGS += -L../datawizard/
CFLAGS += -I../datawizard/

all: tags.o jobs.o workers.o htable.o
	@make -C mechanisms
	@make -C policies

jobs.o: jobs.c
	$(CC) $(CFLAGS) jobs.c -c -o jobs.o

workers.o: workers.c
	$(CC) $(CFLAGS) workers.c -c -o workers.o

tags.o: tags.c tags.h
	$(CC) $(CFLAGS) tags.c -c -o tags.o

htable.o: htable.c htable.h
	$(CC) $(CFLAGS) htable.c -c -o htable.o

clean:
	@make -C policies clean
	@make -C mechanisms clean
	@rm -f *.o
