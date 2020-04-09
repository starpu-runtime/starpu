# GCC compiler
CC=gcc-9
CFLAGS += -O3 -mavx -mfma -fomit-frame-pointer -march=native -ffast-math $(shell pkg-config --cflags starpu-1.3)

LDFLAGS +=$(shell pkg-config --libs starpu-1.3)
EXTERNLIB=extern_tasks.dylib
GENERATEDLIB=generated_tasks.dylib
OBJECTS=$(patsubst %.c,%.o,$(wildcard gen*.c))
LIBPATH=${PWD}/../StarPU.jl/lib

all: ${EXTERNLIB} 

mult: mult.c cpu_mult.o #gpu_mult.o 
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)	

gpu_mult.o: gpu_mult.cu
	nvcc -c $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $^ -o $@

${EXTERNLIB}: cpu_mandelbrot.o
	$(CC) -shared -fPIC $(LDFLAGS) $^ -o $@  

gpu_mult.so: gpu_mult.o
	nvcc $(CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

${GENERATEDLIB}: ${OBJECTS}
	$(CC) -shared -fPIC $(LDFLAGS) $^ -o $@

clean:
	rm *.so *.o *.dylib c_*.genc gencuda_*.cu *.dat



