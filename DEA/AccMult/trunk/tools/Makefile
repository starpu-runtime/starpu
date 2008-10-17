TOOLS=

ifdef USE_FXT
	TOOLS += fxt-tool fxt-stats 
	LDFLAGS+=`pkg-config --libs gtk+-2.0`
endif

ifdef FLASH_RENDER
	DEPS +=  histo-flash.o
	LDFLAGS+= -lming
endif

CFLAGS+= -I../

all: $(TOOLS)

%.o: %.c
	$(CC) $(CFLAGS) $< -c -o $@

fxt-stats: $(DEPS) fxt-stats.o
	$(CC) $(LDFLAGS) $(DEPS) fxt-stats.o -o fxt-stats

fxt-tool: $(DEPS) fxt-tool.o histo-svg.o dag-dot.o gtk-viewer.o histo-paje.o
	$(CC) $(LDFLAGS) $(DEPS) gtk-viewer.o dag-dot.o histo-svg.o fxt-tool.o histo-paje.o -o fxt-tool

gtk-viewer.o: gtk-viewer.c
	$(CC) $(CFLAGS) `pkg-config --cflags gtk+-2.0` gtk-viewer.c -c -o gtk-viewer.o

clean:
	rm -f *.o
	rm -f fxt-tool fxt-stats
