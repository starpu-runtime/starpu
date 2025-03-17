cuda:
	nvcc check_cuda.c -o check_cuda
	./check_cuda

cuda_run:
	nvcc vector.cu -o ./vector
	./vector

opencl:
	gcc check_opencl.c -o check_opencl -lOpenCL
	./check_opencl

hip:
	hipcc $(shell hipconfig --cpp_config) check_hip.c -o check_hip
	./check_hip

hip_run:
	hipcc -std=c++14 $(shell hipconfig --cpp_config) --x cu vector.hip -o vector
	./vector


