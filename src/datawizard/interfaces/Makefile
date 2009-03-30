OBJS := bcsr_interface.o csr_interface.o blas_filters.o blas_interface.o vector_interface.o bcsr_filters.o csr_filters.o vector_filters.o

all: $(OBJS)

ifeq ($(filter ${MAKECMDGOALS},clean distclean),)
%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)
endif

clean:
	@rm -f *.o *.d *.gcno *.gcda
