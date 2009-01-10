CFLAGS+= -Iexternals/scalp/
CFLAGS+= -I../..

OBJS := driver_gordon.o

all: driver_gordon.o

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:   
	@rm -f *.o
