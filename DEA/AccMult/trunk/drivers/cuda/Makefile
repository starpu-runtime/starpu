ifeq ($(CUDA), 1)
	DEPENDENCIES += driver_cuda.o
endif

CFLAGS += -I../../
NVCCFLAGS += -I../../

all: driver_cuda.o

driver_cuda.o: driver_cuda.c driver_cuda.h
	$(NVCC) $(NVCCFLAGS) driver_cuda.c -c -o driver_cuda.o

clean:
	@rm -f *.o *.cu_o *.cubin
