CFLAGS += -I../../

all: driver_core.o

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o 
