STARPU= ../../../starpu.a
CFLAGS+= -I../.. -g

all: example_read

example_read: mm_to_bcsr.o example_read.o mmio.o
	$(CC) example_read.o mm_to_bcsr.o mmio.o -o example_read $(LDFLAGS) $(STARPU)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f example_read
