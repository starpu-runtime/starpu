CFLAGS += -I../../

OBJS := driver_cublas.o

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d *.gcno *.gcda
