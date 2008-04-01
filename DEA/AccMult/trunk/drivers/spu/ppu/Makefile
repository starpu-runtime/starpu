all: mult_spu_ppu.o 

CFLAGS+= -I../../..

mult_spu_ppu.o: mult_spu.h
	$(CC) $(CFLAGS) mult_spu_ppu.c -c -o mult_spu_ppu.o

clean:
	rm -f *.o
