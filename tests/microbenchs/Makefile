export PKG_CONFIG_PATH=../../

LDFLAGS+=$$(pkg-config --libs libstarpu)
CFLAGS+=$$(pkg-config --cflags libstarpu)

all: async-tasks-overhead sync-tasks-overhead

async-tasks-overhead: async-tasks-overhead.o
	$(CC) -o async-tasks-overhead async-tasks-overhead.o $(LDFLAGS)

async-tasks-overhead.o: async-tasks-overhead.c 
	$(CC) $(CFLAGS) async-tasks-overhead.c -c -o async-tasks-overhead.o

sync-tasks-overhead: sync-tasks-overhead.o
	$(CC) -o sync-tasks-overhead sync-tasks-overhead.o $(LDFLAGS)

sync-tasks-overhead.o: sync-tasks-overhead.c 
	$(CC) $(CFLAGS) sync-tasks-overhead.c -c -o sync-tasks-overhead.o

clean:
	rm -f *.o
	rm -f sync-tasks-overhead
	rm -f async-tasks-overhead
