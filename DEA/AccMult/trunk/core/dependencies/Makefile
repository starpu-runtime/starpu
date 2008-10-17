CFLAGS += -I../../

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

all: tags.o htable.o

clean:
	@rm -f *.o *.gcno *.gcda
