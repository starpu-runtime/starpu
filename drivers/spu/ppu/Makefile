all: driver_spu_ppu.o 

CFLAGS+= -I../../..

driver_spu_ppu.o: driver_spu.h
	$(CC) $(CFLAGS) driver_spu_ppu.c -c -o driver_spu_ppu.o

clean:
	@rm -f *.o
