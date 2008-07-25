ifdef USE_FXT
	OPTTARGET+= fxt.o
endif

CFLAGS+= -I../

OBJS := mutex.o threads.o rwlock.o timing.o $(OPTTARGET)

all: $(OBJS)

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d 
