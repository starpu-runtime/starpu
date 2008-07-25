ifdef USE_FXT
	OPTTARGET+= fxt.o
endif

CFLAGS+= -I../

all: threads.o mutex.o rwlock.o timing.o $(OPTTARGET)

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o 
