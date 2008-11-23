STARPU= ../../starpu.a
CFLAGS+= -I../..

all: tag_example

tag_example.o: tag_example.c
	$(CC) $(CFLAGS) tag_example.c -c -o tag_example.o

tag_example: tag_example.o
	$(CC) tag_example.o -o tag_example $(LDFLAGS) $(STARPU)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f tag_example
