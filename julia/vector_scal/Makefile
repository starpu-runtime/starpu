# ICC compiler
#CC =icc
#CFLAGS=-restrict -unroll4 -ipo -falign-loops=256 -O3 -DSTRIDE=${STRIDE} -march=native $(shell pkg-config --cflags starpu-1.3)
# GCC compiler
CC=gcc
CFLAGS += -g -O3 -mavx -fomit-frame-pointer -march=native -ffast-math $(shell pkg-config --cflags starpu-1.3)

LDFLAGS +=$(shell pkg-config --libs starpu-1.3)
EXTERNLIB=extern_tasks.so
GENERATEDLIB=generated_tasks.so
OBJECTS=$(patsubst %.c,%.o,$(wildcard gen*.c))
LIBPATH=${PWD}/../StarPU.jl/lib

all: ${EXTERNLIB}

vector_scal: vector_scal.c cpu_vector_scal.o #gpu_vector_scal.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

gpu_vector_scal.o: gpu_vector_scal.cu
	nvcc -c $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $^ -o $@

${EXTERNLIB}: cpu_vector_scal.c
	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

gpu_vector_scal.so: gpu_vector_scal.o
	nvcc $(CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@ $(LDFLAGS)

cpu_vector_scal_sa: cpu_vector_scal_sa.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

${GENERATEDLIB}: ${OBJECTS}
	$(CC) $(CFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

clean:
	rm -f vector_scal *.so *.o c_*.genc gencuda_*.cu *.dat

# Performance Tests
cstarpu.dat: vector_scal
	STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 ./vector_scal > $@
julia_generatedc.dat:
	LD_LIBRARY_PATH+=${LIBPATH} STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia vector_scal.jl $@
julia_native.dat:
	LD_LIBRARY_PATH+=${LIBPATH} STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia vector_scal_native.jl $@
julia_calllib.dat: ${EXTERNLIB}
	LD_LIBRARY_PATH+=${LIBPATH} JULIA_TASK_LIB="${EXTERNLIB}" STARPU_NOPENCL=0 STARPU_SCHED=dmda STARPU_CALIBRATE=1 julia vector_scal.jl julia_calllib.dat

test: cstarpu.dat julia_generatedc.dat julia_native.dat julia_calllib.dat
