CFLAGS+= -I../

OBJS := blas_model.o

all: $(OBJS)

ifeq ($(filter ${MAKECMDGOALS},clean distclean),)
%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)
endif

clean:
	@rm -f *.o *.d *.gcno *.gcda 
