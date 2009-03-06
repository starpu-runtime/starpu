STARPU= ../../libstarpu.a
CFLAGS+= -I../../src/
CFLAGS+= -I../../

all: dw_mult dw_mult_no_stride dw_mult_no_stride_no_tag

dw_mult.o: dw_mult.c dw_mult.h
	$(CC) $(CFLAGS) dw_mult.c -c -o dw_mult.o

dw_mult_no_stride.o: dw_mult_no_stride.c dw_mult.h
	$(CC) $(CFLAGS) dw_mult_no_stride.c -c -o dw_mult_no_stride.o

dw_mult_no_stride_no_tag.o: dw_mult_no_stride_no_tag.c dw_mult.h
	$(CC) $(CFLAGS) dw_mult_no_stride_no_tag.c -c -o dw_mult_no_stride_no_tag.o

dw_mult: $(STARPU) dw_mult.o
	$(CC) dw_mult.o -o dw_mult $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

dw_mult_no_stride: $(STARPU) dw_mult_no_stride.o
	$(CC) dw_mult_no_stride.o -o dw_mult_no_stride $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

dw_mult_no_stride_no_tag: $(STARPU) dw_mult_no_stride_no_tag.o
	$(CC) dw_mult_no_stride_no_tag.o -o dw_mult_no_stride_no_tag $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f dw_mult dw_mult_no_stride dw_mult_no_stride_no_tag
