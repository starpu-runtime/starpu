.PHONY: ../../../datawizard/spu/spu_coherency.o

all: spu_worker_program.o

SPU_CC=spu-gcc
SPU_EMBED=embedspu
SPU_CFLAGS += -W -Wall -Winline -Wno-main -O3 -g
SPU_INCLUDES =

../../../datawizard/spu/spu_coherency.o: 
	make -C ../../../datawizard/spu/ spu_coherency.o

driver_spu_spu.o: driver_spu_spu.c ../../../datawizard/spu/spu_coherency.o
	$(SPU_CC) $(SPU_LDFLAGS) driver_spu_spu.c ../../../datawizard/spu/spu_coherency.o -o driver_spu_spu.o

spu_worker_program.o: driver_spu_spu.o
	$(SPU_EMBED) spu_worker_program driver_spu_spu.o spu_worker_program.o

clean:
	@rm -f *.o
