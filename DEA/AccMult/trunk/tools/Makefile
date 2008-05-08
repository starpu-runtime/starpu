TOOLS=

ifdef USE_FXT
	TOOLS += fxt-tool 
endif


all: $(TOOLS)

CFLAGS+= -I../

LDFLAGS+= -lming

histo-svg.o: histo-svg.c
	$(CC) $(CFLAGS) histo-svg.c -c -o histo-svg.o

histo-flash.o: histo-flash.c
	$(CC) $(CFLAGS) histo-flash.c -c -o histo-flash.o

fxt-tool: fxt-tool.o histo-flash.o histo-svg.o
	$(CC) $(LDFLAGS) histo-flash.o histo-svg.o fxt-tool.o -o fxt-tool

fxt-tool.o: fxt-tool.c
	$(CC) $(CFLAGS) fxt-tool.c -c -o fxt-tool.o

clean:
	rm -f *.o
	rm -f fxt-tool
