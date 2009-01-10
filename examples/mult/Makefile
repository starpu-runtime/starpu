STARPU= ../../starpu.a
CFLAGS+= -I../..

all: dw_mult

dw_mult.o: dw_mult.c
	$(CC) $(CFLAGS) dw_mult.c -c -o dw_mult.o

dw_mult: $(STARPU) dw_mult.o
	$(CC) dw_mult.o -o dw_mult $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f dw_mult
