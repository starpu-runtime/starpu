ifdef USE_FXT
	OPTTARGET+= fxt.o
endif

CFLAGS+= -I../

all: threads.o mutex.o rwlock.o timing.o $(OPTTARGET)

fxt.o: fxt.h fxt.c
	$(CC) $(CFLAGS) fxt.c -c -o fxt.o

mutex.o: mutex.c mutex.h
	$(CC) $(CFLAGS) mutex.c -c -o mutex.o

rwlock.o: rwlock.c rwlock.h
	$(CC) $(CFLAGS) rwlock.c -c -o rwlock.o

threads.o: threads.c threads.h
	$(CC) $(CFLAGS) threads.c -c -o threads.o

timing.o: timing.c timing.h
	$(CC) $(CFLAGS) timing.c -c -o timing.o

clean:
	@rm -f *.o 
