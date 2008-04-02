all: spu_coherency.o

SPU_CC=spu-gcc
SPU_EMBED=embedspu
SPU_CFLAGS += -W -Wall -Winline -O3 -g
SPU_CFLAGS += -I../../
SPU_CFLAGS += -DUSE_SPU

SPU_INCLUDES =

spu_coherency.o: spu_coherency.c
	$(SPU_CC) $(SPU_CFLAGS) -c -o $@ $^

clean:
	@rm -f *.o
