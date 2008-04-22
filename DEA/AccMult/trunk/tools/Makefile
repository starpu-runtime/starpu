all: fxt-tool

CFLAGS+= -I../

fxt-tool: fxt-tool.o
	$(CC) $(LDFLAGS) fxt-tool.o -o fxt-tool

fxt-tool.o: fxt-tool.c
	$(CC) $(CFLAGS) fxt-tool.c -c -o fxt-tool.o

clean:
	rm -f *.o
	rm -f fxt-tool
