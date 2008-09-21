RIBSCHED= ../../../ribsched.a
CFLAGS+= -I../.. -g

all: example_read

example_read: mm_to_bcsr.o example_read.o mmio.o
	$(CC) example_read.o mm_to_bcsr.o mmio.o -o example_read $(LDFLAGS) $(RIBSCHED)

clean:
	@rm -f *.o *.d
	@rm -f example_read
