STARPU= ../../libstarpu.a

all: tag_example tag_example2

tag_example.o: tag_example.c
	$(CC) $(CFLAGS) tag_example.c -c -o tag_example.o

tag_example2.o: tag_example2.c
	$(CC) $(CFLAGS) tag_example2.c -c -o tag_example2.o

tag_example: $(STARPU) tag_example.o
	$(CC) tag_example.o -o tag_example $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

tag_example2: $(STARPU) tag_example2.o
	$(CC) tag_example2.o -o tag_example2 $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f tag_example tag_example2
