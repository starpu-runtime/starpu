cuda: check_cuda.c
	nvcc check_cuda.c -o check_cuda
	./check_cuda

opencl: check_opencl.c
	gcc check_opencl.c -o check_opencl -lOpenCL
	./check_opencl

HIP_PLATFORM	=	$(shell hipconfig --platform)
hip: check_hip.c
	hipcc $(shell hipconfig --cpp_config) check_hip.c -o check_hip


