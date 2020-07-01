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

all: task_insert_color

task_insert_color: task_insert_color.o
	$(CC) $(CPU_CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) -c -fPIC $(CPU_CFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) -dc $(CUDA_CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

${GENERATEDLIB}: $(C_OBJECTS) $(CUDA_OBJECTS)
	$(LD) -shared $(LDFLAGS) $^ -o $@

PHONY: clean

clean:
	rm -f vector_scal *.so *.o genc_*.c gencuda_*.cu *.dat

# Performance Tests
cstarpu.dat: task_insert_color
	STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 ./task_insert_color > $@
julia_generatedc.dat:
	STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia task_insert_color.jl

test: cstarpu.dat julia_generatedc.dat
