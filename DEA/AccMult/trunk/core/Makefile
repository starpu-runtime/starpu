CFLAGS += -I../

CFLAGS += -I../common/

LDFLAGS += -L../datawizard/
CFLAGS += -I../datawizard/
OBJDEPS += ../datawizard/copy-driver.o ../datawizard/coherency.o

all: tags.o jobs.o workers.o htable.o

jobs.o: jobs.c
	$(CC) $(CFLAGS) jobs.c -c -o jobs.o

workers.o: workers.c
	$(CC) $(CFLAGS) workers.c -c -o workers.o

tags.o: tags.c tags.h
	$(CC) $(CFLAGS) tags.c -c -o tags.o

htable.o: htable.c htable.h
	$(CC) $(CFLAGS) htable.c -c -o htable.o

htable: htable.o

clean:
	@rm -f *.o
