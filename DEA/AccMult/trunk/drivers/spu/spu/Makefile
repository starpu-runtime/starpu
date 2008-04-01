all: spu_worker_program.o

SPU_CC=spu-gcc
SPU_EMBED=embedspu
SPU_CFLAGS += -W -Wall -Winline -Wno-main -O3 -g
SPU_INCLUDES =

mult_spu_spu.o: mult_spu_spu.c
	$(SPU_CC) $(SPU_LDFLAGS) -o $@ $^ $(SPU_LIBS)

spu_worker_program.o: mult_spu_spu.o
	$(SPU_EMBED) spu_worker_program $< $@

clean:
	rm -f *.o
