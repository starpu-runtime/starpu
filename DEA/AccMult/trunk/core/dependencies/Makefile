CFLAGS += -I../../

CFLAGS += -I../../common/

LDFLAGS += -L../../datawizard/
CFLAGS += -I../../datawizard/

all: tags.o htable.o

tags.o: tags.c tags.h
	$(CC) $(CFLAGS) tags.c -c -o tags.o

htable.o: htable.c htable.h
	$(CC) $(CFLAGS) htable.c -c -o htable.o

clean:
	@rm -f *.o
