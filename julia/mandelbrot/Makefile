CC=gcc
CFLAGS += -Wall -Wextra -O3 -mavx -mfma -fomit-frame-pointer -march=native -ffast-math $(shell pkg-config --cflags starpu-1.3)

LDFLAGS +=$(shell pkg-config --libs starpu-1.3) -lm
EXTERNLIB=extern_tasks.so
GENERATEDLIB=generated_tasks.so
#OBJECTS=$(patsubst %.c,%.o,$(wildcard gen*.c))
OBJECTS=$(wildcard gen*.c)
LIBPATH=${PWD}/../StarPU.jl/lib

all: ${EXTERNLIB}

mandelbrot: mandelbrot.c cpu_mandelbrot.o #gpu_mandelbrot.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

gpu_mandelbrot.o: gpu_mandelbrot.cu
	nvcc -c $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $^ -o $@

${EXTERNLIB}: cpu_mandelbrot.c
	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

gpu_mandelbrot.so: gpu_mandelbrot.o
	nvcc $(CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

cpu_mandelbrot_sa: cpu_mandelbrot_sa.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

${GENERATEDLIB}: ${OBJECTS}
	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

clean:
	rm -f mandelbrot *.so *.o c_*.genc gencuda_*.cu *.dat

# Performance Tests
cstarpu.dat: mandelbrot
	STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 ./mandelbrot -0.800671 -0.158392 32 32 4096 4 > $@
julia_generatedc.dat:
	LD_LIBRARY_PATH+=${LIBPATH} STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia mandelbrot.jl $@
julia_native.dat:
	LD_LIBRARY_PATH+=${LIBPATH} STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia mandelbrot_native.jl $@
julia_calllib.dat: ${EXTERNLIB}
	LD_LIBRARY_PATH+=${LIBPATH} JULIA_TASK_LIB="${EXTERNLIB}" STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia mandelbrot.jl julia_calllib.dat

test: cstarpu.dat julia_generatedc.dat julia_native.dat julia_calllib.dat
