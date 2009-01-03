ifdef USE_FXT
	OPTTARGET+= fxt.o
endif

ifdef GOTO
	OPTTARGET+= blas.o
endif

ifdef ATLAS
	OPTTARGET+= blas.o
endif

CFLAGS+= -I../

OBJS := hash.o htable32.o mutex.o threads.o rwlock.o timing.o $(OPTTARGET)

all: $(OBJS)

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d *.gcno *.gcda 
