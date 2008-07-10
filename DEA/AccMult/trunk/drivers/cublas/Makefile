DEPENDENCIES=comp.o threads.o mult.o jobs.o driver_core.o
EXTRADEP=

ifeq ($(CUBLAS), 1)
	DEPENDENCIES += driver_cublas.o
endif

CFLAGS += -I../../

all: driver_cublas.o

driver_cublas.o: driver_cublas.c driver_cublas.h
	$(CC) $(CFLAGS) driver_cublas.c -c -o driver_cublas.o

clean:
	@rm -f *.o *.cu_o *.cubin
