CFLAGS += -I../../

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@
	$(CC) $(CFLAGS) $< -MM > $*.d

all: perfmodel_history.o perfmodel.o

clean:
	@rm -f *.o *.d *.gcno *.gcda
