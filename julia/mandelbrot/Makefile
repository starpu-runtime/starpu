CC=gcc
NVCC=nvcc
ENABLE_CUDA=no
LD=$(CC)

ifeq ($(ENABLE_CUDA),yes)
        LD := ${NVCC}
endif

CFLAGS = -O3 -g $(shell pkg-config --cflags starpu-1.3)
CPU_CFLAGS = ${CFLAGS} -Wall -mavx -fomit-frame-pointer -march=native -ffast-math
CUDA_CFLAGS = ${CFLAGS}
LDFLAGS +=$(shell pkg-config --libs starpu-1.3)

EXTERNLIB=extern_tasks.so
GENERATEDLIB=generated_tasks.so

C_OBJECTS=$(patsubst %.c,%.o,$(wildcard gen*.c))
CUDA_OBJECTS=$(patsubst %.cu,%.o,$(wildcard gen*.cu))
ifneq ($(ENABLE_CUDA),yes)
	CUDA_OBJECTS:=
endif

LIBPATH=${PWD}/../StarPU.jl/lib

all: ${EXTERNLIB}

mandelbrot: mandelbrot.c cpu_mandelbrot.o #gpu_mandelbrot.o
	$(CC) $(CPU_CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) -c -fPIC $(CPU_CFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) -dc $(CUDA_CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

${EXTERNLIB}: cpu_mandelbrot.c
	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

${GENERATEDLIB}: $(C_OBJECTS) $(CUDA_OBJECTS)
	$(LD) -shared $(LDFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f mandelbrot *.so *.o genc_*.c gencuda_*.cu *.dat

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
