TOOLS=

ifdef USE_FXT
	TOOLS += fxt-tool fxt-stats 
endif

ifdef FLASH_RENDER
	DEPS +=  histo-flash.o
	LDFLAGS+= -lming
endif

all: $(TOOLS)

CFLAGS+= -I../


LDFLAGS+=`pkg-config --libs gtk+-2.0`

dag-dot.o: dag-dot.c
	$(CC) $(CFLAGS) dag-dot.c -c -o dag-dot.o

histo-svg.o: histo-svg.c
	$(CC) $(CFLAGS) histo-svg.c -c -o histo-svg.o

histo-flash.o: histo-flash.c
	$(CC) $(CFLAGS) histo-flash.c -c -o histo-flash.o

fxt-stats: $(DEPS) fxt-stats.o
	$(CC) $(LDFLAGS) $(DEPS) fxt-stats.o -o fxt-stats

fxt-stats.o: fxt-stats.c
	$(CC) $(CFLAGS) fxt-stats.c -c -o fxt-stats.o

fxt-tool: $(DEPS) fxt-tool.o histo-svg.o dag-dot.o gtk-viewer.o
	$(CC) $(LDFLAGS) $(DEPS) gtk-viewer.o dag-dot.o histo-svg.o fxt-tool.o -o fxt-tool

fxt-tool.o: fxt-tool.c
	$(CC) $(CFLAGS) fxt-tool.c -c -o fxt-tool.o

gtk-viewer.o: gtk-viewer.c
	$(CC) $(CFLAGS) `pkg-config --cflags gtk+-2.0` gtk-viewer.c -c -o gtk-viewer.o

clean:
	rm -f *.o
	rm -f fxt-tool fxt-stats
