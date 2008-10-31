CFLAGS += -I../../

OBJS := driver_cuda.o

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d *.gcno *.gcda
