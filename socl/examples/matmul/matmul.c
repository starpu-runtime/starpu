/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */
#ifndef STARPU_NON_BLOCKING_DRIVERS
int main(void)
{
	/* testcase does not seem to support blocking drivers */
	return 77;
}
#else

#ifdef __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#define error(...) do { fprintf(stderr, "Error: " __VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#define check(exp) do { err = exp; if(err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error (%d): " #exp "\n", err); exit(EXIT_FAILURE); }} while(0)
#define check2(exp) exp; if(err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error (%d): " #exp "\n", err); exit(EXIT_FAILURE); }
#define check3(exp, err) do { if(err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error (%d): " #exp "\n", err); exit(EXIT_FAILURE); } } while(0)

// Thread block size
#define BLOCK_SIZE 16  // Kernel thread-block size
#define WORK_SIZE 64  // Kernel global size in lines of A (or C)
#define TYPE float

// Basic Matrix dimensions
#define WA (128L * BLOCK_SIZE) // Matrix A width
#ifdef STARPU_QUICK_CHECK
#define HA (128L * BLOCK_SIZE) // Matrix A height
#else
#define HA (512L * BLOCK_SIZE) // Matrix A height
#endif
#define WB (128L * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height
#define BLOCKS (HA / WORK_SIZE)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void printDiff(TYPE*, TYPE*, int, int, int, TYPE);
void computeReference(TYPE*, const TYPE*, const TYPE*, unsigned int, unsigned int, unsigned int);

#define str(x) #x

#define CODE "\
#define TYPE float\n\
__kernel void sgemmNN(int wa, int ha, int wb,  __global TYPE* A, __global TYPE* B, __global TYPE* C) {\n\
#define BS 16\n\
#define BLOCK_SIZE 16\n\
  int bx = get_group_id(0);\n\
  int by = get_group_id(1);\n\
  \n\
  int tx = get_local_id(0);\n\
  int ty = get_local_id(1);\n\
  \n\
  int gx = get_global_id(0);\n\
  int gy = get_global_id(1);\n\
    __local float As[BS][BS+1];\
    __local float Bs[BS][BS+1];\
  \n\
  unsigned int block_w = min(wb - bx * BLOCK_SIZE, BLOCK_SIZE);\n\
  unsigned int block_h = min(ha - by * BLOCK_SIZE, BLOCK_SIZE);\n\
  \n\
  int valid = (gx < wb && gy < ha);\n\
  \n\
  TYPE Csub = (TYPE)0.0;\n\
  \n\
  int pos = 0;\n\
  while (pos < wa) {\n\
    unsigned int size = min(wa-pos, BLOCK_SIZE);\n\
    if (tx < size && gy < ha)\n\
      As[tx][ty] = A[pos + tx + wa * gy];\n\
    if (ty < size && gx < wb)\n\
      Bs[tx][ty] = B[gx + wb * (pos+ty)];\n\
    \n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    \n\
    if (valid) {\n\
      for (int k = 0; k < size; ++k)\n\
        Csub += As[k][ty] * Bs[tx][k];\n\
    }\n\
    pos += size;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
  }\n\
  \n\
  if (valid)\n\
    C[wb * gy + gx] = Csub;\n\
}"

static char * code =  CODE;

int check = 0;

static void __attribute__((unused)) parse_args(int argc, const char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		if (strcmp(argv[i], "-h") == 0)
		{
			printf("usage : %s [-check]\n", argv[0]);
		}
	}
}

// Round Up Division function
size_t roundUp(int group_size, int global_size)
{
	int r = global_size % group_size;
	if(r == 0)
	{
		return global_size;
	}
	else
	{
		return global_size + group_size - r;
	}
}

void fillArray(TYPE* data, int size)
{
	int i;
	const TYPE fScale = (TYPE)(1.0f / (float)RAND_MAX);
	for (i = 0; i < size; ++i)
	{
		data[i] = fScale * rand();
	}
}

void printArray(float* data, int size)
{
	int i;
	for (i = 0; i < size; ++i)
	{
		printf("%d: %.3f\n", i, data[i]);
	}
}

/**
 * Compare two float arrays using L2-norm with an epsilon tolerance for equality
 * @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
 * @param reference  handle to the reference data / gold image
 * @param data       handle to the computed data
 * @param len        number of elements in reference and data
 * @param epsilon    epsilon to use for the comparison
*/
int shrCompareL2fe( const float* reference, const float* data, const unsigned int len, const float epsilon )
{
	assert(epsilon >= 0);

	float error = 0;
	float ref = 0;

	unsigned int i;
	for(i = 0; i < len; ++i)
	{
		float diff = reference[i] - data[i];
		error += diff * diff;
		ref += reference[i] * reference[i];
	}

	float normRef = sqrtf(ref);
	if (fabs(ref) < 1e-7)
	{
#ifdef _DEBUG
		fprintf(stderr, "ERROR, reference l2-norm is 0\n");
#endif
		return 0;
	}
	float normError = sqrtf(error);
	error = normError / normRef;
	int result = error < epsilon;
#ifdef _DEBUG
	if( !result)
	{
		fprintf(stderr, "ERROR, l2-norm error %lf is greater than epsilon %lf \n", error, epsilon);
	}
#endif

	return result;
}


int main(int argc, const char** argv)
{
	cl_uint platform_count;
	cl_platform_id platforms[5];

	cl_int err = CL_SUCCESS;
	unsigned int i, p;

	cl_device_type dev_type = CL_DEVICE_TYPE_ALL;

	void * ptrs[BLOCKS];
	cl_command_queue cqs[BLOCKS];
	cl_mem d_A[BLOCKS];
	cl_mem d_C[BLOCKS];
	cl_mem d_B[BLOCKS];

	cl_event GPUDone[BLOCKS];
	cl_event GPUExecution[BLOCKS];
	struct timeval start, end;

	int workOffset[BLOCKS];
	int workSize[BLOCKS];

	unsigned int sizePerGPU = HC / BLOCKS;
	unsigned int sizeMod = HC % BLOCKS;

	size_t A_size = WA * HA;
	size_t A_mem_size = sizeof(TYPE) * A_size;
	TYPE* A_data;

	size_t B_size = WB * HB;
	size_t B_mem_size = sizeof(TYPE) * B_size;
	TYPE* B_data;

	size_t C_size = WC * HC;
	size_t C_mem_size = sizeof(TYPE) * C_size;
	TYPE* C_data;

	parse_args(argc, argv);

	check(clGetPlatformIDs(5, platforms, &platform_count));
	if (platform_count == 0)
	{
		printf("No platform found\n");
		exit(77);
	}

	cl_uint device_count;
	cl_uint devs[platform_count];
	cl_device_id * devices[platform_count];
	cl_context ctx[platform_count];
	cl_command_queue * commandQueue[platform_count];

	device_count = 0;
	for (p=0; p<platform_count; p++)
	{
		cl_platform_id platform = platforms[p];

		err = clGetDeviceIDs(platform, dev_type, 0, NULL, &devs[p]);
		if (err == CL_DEVICE_NOT_FOUND)
		{
			devs[p] = 0;
			continue;
		}
		if (devs[p] == 0)
		{
		     printf("No OpenCL device found\n");
		     exit(77);
		}
		if (err != CL_SUCCESS)
		{
			fprintf(stderr, "OpenCL Error (%d) in clGetDeviceIDs()\n", err);
			exit(EXIT_FAILURE);
		}
		if (devs[p] == 0)
			continue;

		devices[p] = (cl_device_id*)malloc(sizeof(cl_device_id) * devs[p]);
		commandQueue[p] = (cl_command_queue*)malloc(sizeof(cl_command_queue) * devs[p]);

		check(clGetDeviceIDs(platform, dev_type, devs[p], devices[p], NULL));

		cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
		check2(ctx[p] = clCreateContext(properties, devs[p], devices[p], NULL, NULL, &err));

		for(i = 0; i < devs[p]; ++i)
		{
			cl_device_id device = devices[p][i];
			char name[2048];
			name[0] = '\0';
			clGetDeviceInfo(device, CL_DEVICE_NAME, 2048, name, NULL);
			printf("Device %u: %s\n", i, name);

			commandQueue[p][i] = clCreateCommandQueue(ctx[p], device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
			if (err == CL_INVALID_VALUE)
			{
				fprintf(stderr, "Invalid property for clCreateCommandQueue\n");
				exit(77);
			}
			check3("clCreateCommandQueue", err);
		}

		device_count += devs[p];
	}

	if (device_count == 0)
		error("No device found\n");



	cl_kernel multiplicationKernel[platform_count];

	printf("\nUsing Matrix Sizes: A(%lu x %lu), B(%lu x %lu), C(%lu x %lu)\n",
			(unsigned long)WA, (unsigned long)HA, (unsigned long)WB, (unsigned long)HB, (unsigned long)WC, (unsigned long)HC);

	// allocate host memory for matrices A, B and C
	A_data = (TYPE*)malloc(A_mem_size);
	if (A_data == NULL)
	{
		perror("malloc");
		exit(-1);
	}

	B_data = (TYPE*)malloc(B_mem_size);
	if (B_data == NULL)
	{
		perror("malloc");
		exit(-1);
	}

	C_data = (TYPE*) malloc(C_mem_size);
	if (C_data == NULL)
	{
		perror("malloc");
		exit(-1);
	}

	cl_program program[platform_count];

	for (p=0; p<platform_count; p++)
	{
		if (devs[p] == 0)
			continue;

		check2(program[p] = clCreateProgramWithSource(ctx[p], 1, (const char **)&code, NULL, &err));

		check(clBuildProgram(program[p], 0, NULL, NULL, NULL, NULL));

		check2(multiplicationKernel[p] = clCreateKernel(program[p], "sgemmNN", &err));
	}

	printf("Initializing data...\n");
	srand(2008);
	fillArray(A_data, A_size);
	fillArray(B_data, B_size);
	memset(C_data, 0, C_size);


	printf("Computing...\n");
	workOffset[0] = 0;
	gettimeofday(&start, NULL);

	size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
	int c = 0;
	for (p=0; p<platform_count;p++)
	{
		for (i=0; i<devs[p]; i++)
		{
			check2(d_B[c] = clCreateBuffer(ctx[p], CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, HB * WB * sizeof(TYPE), B_data, &err));
			c++;
		}
	}

	for(i=0; i < BLOCKS; ++i)
	{
		int d = i % device_count;
		cl_uint platform = 0;

		// determine device platform
		int dev = d;
		for (platform = 0; platform < platform_count; platform++)
		{
			if ((cl_int)(dev - devs[platform]) < 0)
				break;
			dev -= devs[platform];
		}
		assert(platform < platform_count);

		workSize[i] = (i < sizeMod) ? sizePerGPU+1 : sizePerGPU;

		check2(d_A[i] = clCreateBuffer(ctx[platform], CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, workSize[i] * WA * sizeof(TYPE), &A_data[workOffset[i] * WA], &err));
		check2(d_C[i] = clCreateBuffer(ctx[platform], CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, workSize[i] * WC * sizeof(TYPE), &C_data[workOffset[i] * WC], &err));

		check(clSetKernelArg(multiplicationKernel[platform], 0, sizeof(cl_int), &workSize[i]));
		check(clSetKernelArg(multiplicationKernel[platform], 1, sizeof(cl_int), &workSize[i]));
		check(clSetKernelArg(multiplicationKernel[platform], 2, sizeof(cl_int), &workSize[i]));
		check(clSetKernelArg(multiplicationKernel[platform], 3, sizeof(cl_mem), (void *) &d_A[i]));
		check(clSetKernelArg(multiplicationKernel[platform], 4, sizeof(cl_mem), (void *) &d_B[d]));
		check(clSetKernelArg(multiplicationKernel[platform], 5, sizeof(cl_mem), (void *) &d_C[i]));

		size_t globalWorkSize[] = {roundUp(BLOCK_SIZE,WC), roundUp(BLOCK_SIZE,workSize[i])};

		check(clEnqueueNDRangeKernel(commandQueue[platform][dev], multiplicationKernel[platform], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &GPUExecution[i]));

		// Non-blocking copy of result from device to host
		cqs[i] = commandQueue[platform][dev];
		check2(ptrs[i] = clEnqueueMapBuffer(cqs[i], d_C[i], CL_FALSE, CL_MAP_READ, 0, WC * sizeof(TYPE) * workSize[i], 1, &GPUExecution[i], &GPUDone[i], &err));

		if(i+1 < BLOCKS)
			workOffset[i + 1] = workOffset[i] + workSize[i];
	}


	// CPU sync with GPU
	for (p=0; p<platform_count;p++)
	{
		cl_uint dev;
		for (dev=0; dev<devs[p]; dev++)
		{
			clFinish(commandQueue[p][dev]);
		}
	}

	gettimeofday(&end, NULL);
	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	double dSeconds = timing/1000/1000;
	double dNumOps = 2.0 * (double)WA * (double)HA * (double)WB;
	double gflops = 1.0e-9 * dNumOps/dSeconds;

	printf("Throughput = %.4f GFlops/s, Time = %.5f s, Size = %.0f, NumDevsUsed = %d, Blocks = %ld, Workgroup = %zu\n",
			gflops, dSeconds, dNumOps, device_count, BLOCKS, localWorkSize[0] * localWorkSize[1]);

	// compute reference solution
	if (check)
	{
		printf("Comparing results with CPU computation... ");
		TYPE* reference = (TYPE*)malloc(C_mem_size);
		computeReference(reference, A_data, B_data, HA, WA, WB);

		// check result
		int res = shrCompareL2fe(reference, C_data, C_size, 1.0e-6f);
		if (res == 0)
		{
			printf("\n\n");
			printDiff(reference, C_data, WC, HC, 100, 1.0e-5f);
		}
		else printf("PASSED\n\n");
		free(reference);
	}

	for(i = 0; i < BLOCKS; i++)
	{
		clEnqueueUnmapMemObject(cqs[i], d_C[i], ptrs[i], 0, NULL, NULL);
	}

	for(i = 0; i < BLOCKS; i++)
	{
		clFinish(cqs[i]);
	}

	for (i=0; i<device_count; i++)
	{
		clReleaseMemObject(d_B[i]);
	}

	for(i = 0; i < BLOCKS; i++)
	{
		clReleaseMemObject(d_A[i]);
		clReleaseMemObject(d_C[i]);
		clReleaseEvent(GPUExecution[i]);
		clReleaseEvent(GPUDone[i]);
	}


	for (p=0; p<platform_count;p++)
	{
		if (devs[p] == 0)
			continue;

		check(clReleaseKernel(multiplicationKernel[p]));
		check(clReleaseProgram(program[p]));
		check(clReleaseContext(ctx[p]));
		cl_uint k;
		for(k = 0; k < devs[p]; ++k)
		{
			check(clReleaseCommandQueue(commandQueue[p][k]));
		}
	}

	free(A_data);
	free(B_data);
	free(C_data);

	return 0;
}

void printDiff(TYPE *data1, TYPE *data2, int width, int height, int listLength, TYPE listTol)
{
	printf("Listing first %d Differences > %.6f...\n", listLength, listTol);
	int i,j,k;
	int error_count=0;
	for (j = 0; j < height; j++)
	{
		if (error_count < listLength)
		{
			printf("\n  Row %d:\n", j);
		}
		for (i = 0; i < width; i++)
		{
			k = j * width + i;
			float diff = fabs(data1[k] - data2[k]);
			if (diff > listTol)
			{
				if (error_count < listLength)
				{
					printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], diff);
				}
				error_count++;
			}
		}
	}
	printf(" \n  Total Errors = %d\n\n", error_count);
}

/**
 * Compute reference data set
 * C = A * B
 * @param C          reference data, computed but preallocated
 * @param A          matrix A as provided to device
 * @param B          matrix B as provided to device
 * @param hA         height of matrix A
 * @param wB         width of matrix B
*/
void computeReference(TYPE* C, const TYPE* A, const TYPE* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	unsigned int i,j,k;
	for (i = 0; i < hA; ++i)
		for (j = 0; j < wB; ++j)
		{
			double sum = 0;
			for (k = 0; k < wA; ++k)
			{
				double a = A[i * wA + k];
				double b = B[k * wB + j];
				sum += a * b;
			}
			C[i * wB + j] = (TYPE)sum;
		}
}
#endif /* STARPU_NON_BLOCKING_DRIVERS */
