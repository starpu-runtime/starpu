/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011 University of Bordeaux
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

#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#define error(...) do { fprintf(stderr, "Error: " __VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#define check(exp) do { cl_int err = exp; if(err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error (%d): " #exp "\n", err); exit(EXIT_FAILURE); }} while(0)
#define check2(exp) exp; if(err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error (%d): " #exp "\n", err); exit(EXIT_FAILURE); }

#ifdef UNUSED
#elif defined(__GNUC__)
#define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#else
#define UNUSED(x) x
#endif

// Thread block size
#define BLOCK_SIZE 16  // Kernel thread-block size
#define WORK_SIZE 64  // Kernel global size in lines of A (or C)
#define TYPE float

// Basic Matrix dimensions
#define WA (1024L * BLOCK_SIZE) // Matrix A width
#define HA (512L * BLOCK_SIZE) // Matrix A height
#define WB (1024L * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height
#define BLOCKS (HA / WORK_SIZE)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void printDiff(TYPE*, TYPE*, int, int, int, TYPE);
void computeGold(TYPE*, const TYPE*, const TYPE*, unsigned int, unsigned int, unsigned int);

#define str(x) #x

#define CODE "\n\
#define BS 16\n\
\n\
__kernel void matrixMul(\n\
	 const int N,\n\
	 const int P,\n\
	 const int M,\n\
	 __global float* A,\n\
	 __global float* B, \n\
	 __global float* C) {\n\
    int row = get_global_id(1); \n\
    int col = get_global_id(0); \n\
    float sum = 0.0f;\n\
    float sum2 = 0.0f;\n\
    int x = get_local_id(0);\n\
    int y = get_local_id(1);\n\
    __local float atile[BS][BS+1];\n\
    __local float btile[BS][BS+1];\n\
    for (int t=0; t<N; t+=BS) {\n\
        atile[y][x] = A[t + row * BS + x] ;\n\
    	btile[y][x] = B[(t + y) *N + col];\n\
    	barrier(CLK_LOCAL_MEM_FENCE);\n\
	for (int i=0; i<BS; i++) {\n\
	    sum += atile[y][i] * btile[i][x];\n\
	}	    \n\
    	barrier(CLK_LOCAL_MEM_FENCE);\n\
    }\n\
    C[row*N+col] = sum + sum2;\n\
\n\
}\n\
\n\
__kernel void saxpy( float a, __local float *b, float *c )\n\
{\n\
	c[0] += a*b[0];\n\
	c[1] += a*b[1];\n\
	c[2] += a*b[2];\n\
	c[3] += a*b[3];\n\
	c[4] += a*b[4];\n\
	c[5] += a*b[5];\n\
	c[6] += a*b[6];\n\
	c[7] += a*b[7];\n\
	c[8] += a*b[8];\n\
	c[9] += a*b[9];\n\
	c[10] += a*b[10];\n\
	c[11] += a*b[11];\n\
	c[12] += a*b[12];\n\
	c[13] += a*b[13];\n\
	c[14] += a*b[14];\n\
	c[15] += a*b[15];\n\
}\n\
\n\
\n\
__kernel void sgemmNN( int lda,int ldb, int ldc, __global float *A, __global float *B, __global float* C)\n\
{\n\
	const int inx = get_local_id(0);\n\
	const int iny = get_local_id(1);\n\
	const int ibx = get_group_id(0) * 16;\n\
	const int iby = get_group_id(1) * 16;\n\
	const int id = inx + iny*16;\n\
	int i;\n\
	\n\
	long Aidx = ibx + id;\n\
	long Bidx = inx + mul24( iby + iny, ldb );\n\
	long Cidx = ibx + id  + mul24( iby, ldc );\n\
	\n\
	long Blast = ldb;\n\
	\n\
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};\n\
    \n\
	do\n\
	{\n\
		float a[4] = { A[Aidx+0*lda], A[Aidx+1*lda], A[Aidx+2*lda], A[Aidx+3*lda] };\n\
\n\
		__local float bs[16][17];\n\
\n\
		bs[inx][iny]    = B[Bidx+0*ldb];\n\
		bs[inx][iny+4]  = B[Bidx+4*ldb];\n\
		bs[inx][iny+8]  = B[Bidx+8*ldb];\n\
		bs[inx][iny+12] = B[Bidx+12*ldb];\n\
		Bidx+= 16;\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
		Aidx += 4*lda;\n\
		saxpy( a[0], &bs[0][0], c );		a[0] = A[Aidx+0*lda];\n\
		saxpy( a[1], &bs[1][0], c );		a[1] = A[Aidx+1*lda];\n\
		saxpy( a[2], &bs[2][0], c );		a[2] = A[Aidx+2*lda];\n\
		saxpy( a[3], &bs[3][0], c );		a[3] = A[Aidx+3*lda];\n\
	} while( Bidx < Blast );\n\
\n\
	for(i = 0; i < 16; i++, Cidx += ldc )\n\
		C[Cidx] = c[i];\n\
\n\
}"

#define CODE2 "\
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

static char * code =  CODE2;

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

double executionTime(cl_event event)
{
  cl_ulong start, end;

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

  return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

#define shrLog(...) fprintf(stderr, __VA_ARGS__);

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size)
{
  int r = global_size % group_size;
  if(r == 0)
  {
    return global_size;
  } else
  {
    return global_size + group_size - r;
  }
}

// Helper function to init data arrays 
// *********************************************************************
void fillArray(TYPE* pfData, int iSize)
{
  int i;
  const TYPE fScale = (TYPE)(1.0f / (float)RAND_MAX);
  for (i = 0; i < iSize; ++i)
  {
    pfData[i] = fScale * rand();
  }
}

// Helper function to print data arrays 
// *********************************************************************
void shrPrintArray(float* pfData, int iSize)
{
  int i;
  for (i = 0; i < iSize; ++i)
  {
    shrLog("%d: %.3f\n", i, pfData[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays using L2-norm with an epsilon tolerance for equality
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
int shrCompareL2fe( const float* reference, const float* data,
    const unsigned int len, const float epsilon )
{
  assert(epsilon >= 0);

  float error = 0;
  float ref = 0;

  unsigned int i;
  for(i = 0; i < len; ++i) {
    float diff = reference[i] - data[i];
    error += diff * diff;
    ref += reference[i] * reference[i];
  }

  float normRef = sqrtf(ref);
  if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
    fprintf(stderr, "ERROR, reference l2-norm is 0\n");
#endif
    return 0;
  }
  float normError = sqrtf(error);
  error = normError / normRef;
  int result = error < epsilon;
#ifdef _DEBUG
  if( ! result)
  {
    fprintf(stderr, "ERROR, l2-norm error %d is greater than epsilon %lf \n", error, epsilon);
  }
#endif

  return result;
}


int main(int UNUSED(argc), const char** UNUSED(argv))
{
  cl_uint platform_count;
  cl_platform_id platforms[5];

  cl_int err = CL_SUCCESS;
  unsigned int i, p;

  cl_device_type dev_type = CL_DEVICE_TYPE_ALL;

   /* Get platforms */
  check(clGetPlatformIDs(5, platforms, &platform_count));
  if (platform_count == 0)
    error("No platform found\n");

  cl_uint device_count;
   cl_uint devs[platform_count];
   cl_device_id * devices[platform_count];
   cl_context ctx[platform_count];
   cl_command_queue * commandQueue[platform_count];

   device_count = 0;
   for (p=0; p<platform_count; p++) {
      cl_platform_id platform = platforms[p];

      cl_int err = clGetDeviceIDs(platform, dev_type, 0, NULL, &devs[p]);
      if (err == CL_DEVICE_NOT_FOUND) {
        devs[p] = 0;
        continue;
      }
      check(err);
      if (devs[p] == 0)
         continue;

      devices[p] = (cl_device_id*)malloc(sizeof(cl_device_id) * devs[p]);
      commandQueue[p] = (cl_command_queue*)malloc(sizeof(cl_command_queue) * devs[p]);

      check(clGetDeviceIDs(platform, dev_type, devs[p], devices[p], NULL));

      cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
      check2(ctx[p] = clCreateContext(properties, devs[p], devices[p], NULL, NULL, &err));

      for(i = 0; i < devs[p]; ++i) 
      {
         // get and print the device for this queue
         cl_device_id device = devices[p][i];
         char name[2048];
         name[0] = '\0';
         clGetDeviceInfo(device, CL_DEVICE_NAME, 2048, name, NULL);
         printf("Device %d: %s\n", i, name);

         // create command queue
         check2(commandQueue[p][i] = clCreateCommandQueue(ctx[p], device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
      }

      device_count += devs[p];
   }

   if (device_count == 0)
      error("No device found\n");



   cl_kernel multiplicationKernel[platform_count];


  // Optional Command-line multiplier for matrix sizes
  printf("\nUsing Matrix Sizes: A(%lu x %lu), B(%lu x %lu), C(%lu x %lu)\n", 
      (unsigned long)WA, (unsigned long)HA, (unsigned long)WB, (unsigned long)HB, (unsigned long)WC, (unsigned long)HC);

  // allocate host memory for matrices A, B and C
  size_t A_size = WA * HA;
  size_t A_mem_size = sizeof(TYPE) * A_size;
  TYPE* A_data = (TYPE*)malloc(A_mem_size);
  if (A_data == NULL) {
    perror("malloc");
  }

  size_t B_size = WB * HB;
  size_t B_mem_size = sizeof(TYPE) * B_size;
  TYPE* B_data = (TYPE*)malloc(B_mem_size);
  if (B_data == NULL) {
    perror("malloc");
  }

  size_t C_size = WC * HC;
  size_t C_mem_size = sizeof(TYPE) * C_size;
  TYPE* C_data = (TYPE*) malloc(C_mem_size);
  if (C_data == NULL) {
    perror("malloc");
  }

  cl_program program[platform_count];

  for (p=0; p<platform_count; p++) {
     if (devs[p] == 0)
        continue;

     check2(program[p] = clCreateProgramWithSource(ctx[p], 1, (const char **)&code, NULL, &err));

     check(clBuildProgram(program[p], 0, NULL, NULL, NULL, NULL));

     // Create Kernel
     check2(multiplicationKernel[p] = clCreateKernel(program[p], "sgemmNN", &err));
  }

  // initialize host memory
  printf("Initializing data...\n");
  srand(2008);
  fillArray(A_data, A_size);
  fillArray(B_data, B_size);
   memset(C_data, 0, C_size);


  printf("Computing...\n");

  // Run multiplication on 1..deviceCount GPUs to compare improvement
  printf("\nRunning Computations on 1 - %d GPU's...\n\n", device_count);
  {
    cl_mem d_A[BLOCKS];
    cl_mem d_C[BLOCKS];
    void * ptrs[BLOCKS];
    cl_mem d_B[BLOCKS];

    cl_event GPUDone[BLOCKS];
    cl_event GPUExecution[BLOCKS];

    unsigned int sizePerGPU = HC / BLOCKS;
    unsigned int sizeMod = HC % BLOCKS;
    if (HC < BLOCKS) {
      printf("Not enough data to split in %d blocks. Test skipped\n", device_count);
      exit(0);
    }

    int workOffset[BLOCKS];
    int workSize[BLOCKS];

    workOffset[0] = 0;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    int c = 0;
    for (p=0; p<platform_count;p++) {
       for (i=0; i<devs[p]; i++) {
          check2(d_B[c] = clCreateBuffer(ctx[p], CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, HB * WB * sizeof(TYPE), B_data, &err));
          c++;
       }
    }

    unsigned int i;
    for(i=0; i < BLOCKS; ++i) 
    {
      int d = i % device_count;
      cl_uint p = 0;
      // determine device platform
      int dev = d;
      for (p = 0; p < platform_count; p++) {
         if ((cl_int)(dev - devs[p]) < 0)
            break;
         dev -= devs[p];
      }

      // Input buffer
      workSize[i] = (i < sizeMod) ? sizePerGPU+1 : sizePerGPU;        

      check2(d_A[i] = clCreateBuffer(ctx[p], CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, workSize[i] * WA * sizeof(TYPE), &A_data[workOffset[i] * WA], &err));
      check2(d_C[i] = clCreateBuffer(ctx[p], CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, workSize[i] * WC * sizeof(TYPE), &C_data[workOffset[i] * WC], &err));

      // set the args values
      check(clSetKernelArg(multiplicationKernel[p], 0, sizeof(cl_int), &workSize[i]));
      check(clSetKernelArg(multiplicationKernel[p], 1, sizeof(cl_int), &workSize[i]));
      check(clSetKernelArg(multiplicationKernel[p], 2, sizeof(cl_int), &workSize[i]));
      check(clSetKernelArg(multiplicationKernel[p], 3, sizeof(cl_mem), (void *) &d_A[i]));
      check(clSetKernelArg(multiplicationKernel[p], 4, sizeof(cl_mem), (void *) &d_B[d]));
      check(clSetKernelArg(multiplicationKernel[p], 5, sizeof(cl_mem), (void *) &d_C[i]));


      // Multiplication - non-blocking execution:  launch and push to device(s)
      size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE,WC), shrRoundUp(BLOCK_SIZE,workSize[i])};

      check(clEnqueueNDRangeKernel(commandQueue[p][dev], multiplicationKernel[p], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &GPUExecution[i]));

      // Non-blocking copy of result from device to host
      //check2(ptrs[i] = clEnqueueMapBuffer(commandQueue[p][dev], d_C[i], CL_FALSE, CL_MAP_READ, 0, WC * sizeof(TYPE) * workSize[i], 1, &GPUExecution[i], &GPUDone[i], &err));

      if(i+1 < BLOCKS)
        workOffset[i + 1] = workOffset[i] + workSize[i];
    }


    // CPU sync with GPU
    for (p=0; p<platform_count;p++) {
      cl_uint dev;
      for (dev=0; dev<devs[p]; dev++) {
         clFinish(commandQueue[p][dev]);
      }
    }

    gettimeofday(&end, NULL);
    double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

    //double dSeconds = shrDeltaT(0)/(double)nIter;
    double dSeconds = timing/1000/1000;
    double dNumOps = 2.0 * (double)WA * (double)HA * (double)WB;
    double gflops = 1.0e-9 * dNumOps/dSeconds;

#ifdef SHORT_LOG
    printf("%f\n", timing/1000.0);
#else
    printf("Throughput = %.4f GFlops/s, Time = %.5f s, Size = %.0f, NumDevsUsed = %d, Blocks = %ld, Workgroup = %zu\n", 
        gflops, dSeconds, dNumOps, device_count, BLOCKS, localWorkSize[0] * localWorkSize[1]);

    // Print kernel timing per GPU
    /*for(i = 0; i < k; i++) 
      printf("  Kernel execution time on GPU %d \t: %.5f s\n", i, executionTime(GPUExecution[i]));
      printf("\n");*/
#endif

    
    for (i=0; i<device_count; i++) {
      clReleaseMemObject(d_B[i]);
    }

    // Release mem and event objects    
    for(i = 0; i < BLOCKS; i++) 
    {
      clReleaseMemObject(d_A[i]);
      clReleaseMemObject(d_C[i]);
      clReleaseEvent(GPUExecution[i]);
      //clReleaseEvent(GPUDone[i]);
    }
  }


  // compute reference solution

#ifdef CHECK
  printf("Comparing results with CPU computation... ");
  TYPE* reference = (TYPE*)malloc(C_mem_size);
  computeGold(reference, A_data, B_data, HA, WA, WB);

  // check result
  int res = shrCompareL2fe(reference, C_data, C_size, 1.0e-6f);
  if (res == 0) 
  {
    printf("\n\n");
    printDiff(reference, C_data, WC, HC, 100, 1.0e-5f);
  }
  else printf("PASSED\n\n");
  free(reference);
#endif

  // clean up OCL resources

  for (p=0; p<platform_count;p++) {
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

  // clean up memory
  free(A_data);
  free(B_data);
  free(C_data);

  return 0;
}

void printDiff(TYPE *data1, TYPE *data2, int width, int height, int iListLength, TYPE fListTol)
{
  shrLog("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
  int i,j,k;
  int error_count=0;
  for (j = 0; j < height; j++) 
  {
    if (error_count < iListLength)
    {
      shrLog("\n  Row %d:\n", j);
    }
    for (i = 0; i < width; i++) 
    {
      k = j * width + i;
      float fDiff = fabs(data1[k] - data2[k]);
      if (fDiff > fListTol) 
      {                
        if (error_count < iListLength)
        {
          shrLog("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
        }
        error_count++;
      }
    }
  }
  shrLog(" \n  Total Errors = %d\n\n", error_count);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
  void
computeGold(TYPE* C, const TYPE* A, const TYPE* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
  unsigned int i,j,k;
  for (i = 0; i < hA; ++i)
    for (j = 0; j < wB; ++j) {
      double sum = 0;
      for (k = 0; k < wA; ++k) {
        double a = A[i * wA + k];
        double b = B[k * wB + j];
        sum += a * b;
      }
      C[i * wB + j] = (TYPE)sum;
    }
}

