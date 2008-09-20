RIBSCHED= ../../../ribsched.a
CFLAGS+= -I../.. -g

all: example_read mm_to_bcsr

example_read: example_read.o mmio.o
	$(CC) example_read.o mmio.o -o example_read $(LDFLAGS) $(RIBSCHED)

mm_to_bcsr: mm_to_bcsr.o
	$(CC) mm_to_bcsr.o mmio.o -o mm_to_bcsr $(LDFLAGS) $(RIBSCHED)

clean:
	@rm -f *.o *.d
	@rm -f example_read mm_to_bcsr
