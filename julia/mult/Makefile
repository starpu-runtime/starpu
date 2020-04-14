# tile size. Should be changed in mult.jl as well
STRIDE=72

# ICC compiler
#CC =icc
#CFLAGS=-restrict -unroll4 -ipo -falign-loops=256 -O3 -DSTRIDE=${STRIDE} -march=native $(shell pkg-config --cflags starpu-1.3)
# GCC compiler
CC=gcc
CFLAGS += -O3 -DSTRIDE=${STRIDE} -mavx -fomit-frame-pointer -march=native -ffast-math $(shell pkg-config --cflags starpu-1.3)

LDFLAGS +=$(shell pkg-config --libs starpu-1.3)
EXTERNLIB=extern_tasks.so
GENERATEDLIB=generated_tasks.so
#OBJECTS=$(patsubst %.c,%.o,$(wildcard gen*.c))
OBJECTS=$(wildcard gen*.c)
LIBPATH=${PWD}/../StarPU.jl/lib

all: ${EXTERNLIB}

mult: mult.c cpu_mult.o #gpu_mult.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

gpu_mult.o: gpu_mult.cu
	nvcc -c $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $^ -o $@

${EXTERNLIB}: cpu_mult.c
	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

gpu_mult.so: gpu_mult.o
	nvcc $(CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

cpu_mult_sa: cpu_mult_sa.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

${GENERATEDLIB}: ${OBJECTS}
	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

clean:
	rm -f mult *.so *.o c_*.genc gencuda_*.cu *.dat

# Performance Tests
cstarpu.dat: mult
	STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 ./mult > $@
julia_generatedc.dat:
	LD_LIBRARY_PATH+=${LIBPATH} STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia mult.jl $@
julia_native.dat:
	LD_LIBRARY_PATH+=${LIBPATH} STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia mult_native.jl $@
julia_calllib.dat: ${EXTERNLIB}
	LD_LIBRARY_PATH+=${LIBPATH} JULIA_TASK_LIB="${EXTERNLIB}" STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia mult.jl julia_calllib.dat

test: cstarpu.dat julia_generatedc.dat julia_native.dat julia_calllib.dat
