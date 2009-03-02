CFLAGS += -I../../

OBJS := driver_core.o

all: $(OBJS)

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d *.gcno *.gcda 
