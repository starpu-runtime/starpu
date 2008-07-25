CFLAGS+= -I../../

OBJS := csr_interface.o blas_filters.o blas_interface.o vector_interface.o csr_filters.o vector_filters.o

all: $(OBJS)

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@rm -f *.o *.d
