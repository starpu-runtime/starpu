





CC = gcc
CFLAGS += $(shell pkg-config --cflags starpu-1.3)
LDFLAGS += $(shell pkg-config --libs starpu-1.3)


all: libjlstarpu_c_wrapper.so build/mult build/extern_tasks.so build/generated_tasks.so



libjlstarpu_c_wrapper.so: ../src/Wrapper/C/jlstarpu_task_submit.c ../src/Wrapper/C/jlstarpu_simple_functions.c ../src/Wrapper/C/jlstarpu_data_handles.c
	$(CC) -O3 -shared -fPIC $(CFLAGS) $^ -o $@ $(LDFLAGS)



build/mult: mult.c build/cpu_mult.o build/gpu_mult.o
	$(CC) $(CFLAGS) -O3 $^ -o $@ $(LDFLAGS)	

build/gpu_mult.o: gpu_mult.cu
	nvcc -c -O3 $(CFLAGS) $^ -o $@

build/cpu_mult.o: cpu_mult.c
	$(CC) -c $(CFLAGS) -O3 $^ -o $@ $(LDFLAGS)




build/extern_tasks.so: build/cpu_mult.so build/gpu_mult.so
	gcc -shared -fPIC $^ -o $@

build/cpu_mult.so: cpu_mult.c
	$(CC) -O3 -shared -fPIC $(CFLAGS) $^ -o $@ $(LDFLAGS)

build/gpu_mult.so: gpu_mult.cu
	nvcc -O3 $(CFLAGS) $^ --shared --compiler-options '-fPIC' -o $@




build/generated_tasks.so: cpu_cuda_mult.jl
	julia $^




clean:
	rm build/* libjlstarpu_c_wrapper.so
