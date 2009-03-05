CFLAGS+= -Iexternals/scalp/
CFLAGS+= -I../..

OBJS := driver_gordon.o

all: driver_gordon.o

ifeq ($(filter ${MAKECMDGOALS},clean distclean),)
%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)
endif

clean:   
	@rm -f *.o *.d
