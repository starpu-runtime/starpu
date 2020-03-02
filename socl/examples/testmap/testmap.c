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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define error(...) do { fprintf(stderr, "Error: " __VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#define check(err, str) do { if(err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error (%d): %s\n",err, str); exit(EXIT_FAILURE); }} while(0)

#ifdef UNUSED
#elif defined(__GNUC__)
# define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#else
# define UNUSED(x) x
#endif

#define SIZE 1024
#define TYPE float
#define REALSIZE (SIZE * sizeof(TYPE))

const char * kernel_src = "__kernel void add(__global float*s1, __global float*s2, __global float*d) { \
   size_t x = get_global_id(0);\n\
   size_t y = get_global_id(1);\n\
   size_t w = get_global_size(0); \n\
   int idx = y*w+x; \n\
#ifdef SOCL_DEVICE_TYPE_GPU \n\
   d[idx] = s1[idx] + s2[idx];\n\
#endif \n\
#ifdef SOCL_DEVICE_TYPE_CPU \n\
   d[idx] = s1[idx] + 2* s2[idx];\n\
#endif \n\
#ifdef SOCL_DEVICE_TYPE_ACCELERATOR \n\
   d[idx] = s1[idx] + 3 * s2[idx];\n\
#endif \n\
#ifdef SOCL_DEVICE_TYPE_UNKNOWN \n\
   d[idx] = s1[idx] + 4 * s2[idx];\n\
#endif \n\
}";



int main(int UNUSED(argc), char** UNUSED(argv)) {
   cl_platform_id platforms[15];
   cl_uint num_platforms;
   cl_device_id devices[15];
   cl_uint num_devices;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_mem s1m, s2m, dm;
   cl_command_queue cq;
   cl_int err;
   unsigned int i;

   TYPE * s1, *s2, d[SIZE];

   printf("Querying platform...\n");
   clGetPlatformIDs(0, NULL, &num_platforms);
   if (num_platforms == 0) {
      printf("No OpenCL platform found.\n");
      exit(77);
   }

   err = clGetPlatformIDs(sizeof(platforms)/sizeof(cl_platform_id), platforms, &num_platforms);
   check(err, "clGetPlatformIDs");

   int platform_idx = -1;
   for (i=0; i<num_platforms;i++) {
    char vendor[256];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    if (strcmp(vendor, "Inria") ==  0) {
      platform_idx = i;
    }
  }

  if (platform_idx == -1) {
      printf("SOCL platform not found.\n");
      exit(77);
  }


   printf("Querying devices...\n");
   err = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, sizeof(devices)/sizeof(cl_device_id), devices, &num_devices);
   check(err, "clGetDeviceIDs");

   if (num_devices == 0) {
      printf("No OpenCL device found\n");
      exit(77);
   }

   printf("Creating context...\n");
   cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platform_idx], 0};
   context = clCreateContext(properties, num_devices, devices, NULL, NULL, &err);
   check(err, "clCreateContext");

   printf("Creating program...\n");
   program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, &err);
   check(err, "clCreateProgram");

   printf("Building program...\n");
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   check(err, "clBuildProgram");

   printf("Creating kernel...\n");
   kernel = clCreateKernel(program, "add", &err);
   check(err, "clCreateKernel");

   printf("Creating buffers...\n");
   s1m = clCreateBuffer(context, CL_MEM_READ_WRITE, REALSIZE, NULL, &err);
   check(err, "clCreateBuffer s1");
   s2m = clCreateBuffer(context, CL_MEM_READ_ONLY, REALSIZE, NULL, &err);
   check(err, "clCreateBuffer s2");
   dm = clCreateBuffer(context, CL_MEM_WRITE_ONLY, REALSIZE, NULL, &err);
   check(err, "clCreateBuffer d");

   printf("Creating command queue...\n");
   cl_event eventK, eventR;

#ifdef PROFILING
   cq = clCreateCommandQueue(context, NULL, CL_QUEUE_PROFILING_ENABLE, &err);
#else
   cq = clCreateCommandQueue(context, NULL, 0, &err);
#endif
   check(err, "clCreateCommandQueue");

   printf("Enqueueing MapBuffer...\n");
   s1 = clEnqueueMapBuffer(cq, s1m, CL_TRUE, CL_MAP_WRITE, 0, REALSIZE, 0, NULL, NULL, &err);
   check(err, "clEnqueueMapBuffer s1");
   s2 = clEnqueueMapBuffer(cq, s2m, CL_TRUE, CL_MAP_WRITE, 0, REALSIZE, 0, NULL, NULL, &err);
   check(err, "clEnqueueMapBuffer s2");

   {
      for (i=0; i<SIZE; i++) {
         s1[i] = 2.0;
         s2[i] = 7.0;
         d[i] = 98.0;
      }
   }

   printf("Enqueueing UnmapMemObject...\n");
   err = clEnqueueUnmapMemObject(cq, s1m, s1, 0, NULL, NULL);
   check(err, "clEnqueueUnmapMemObject s1");
   err = clEnqueueUnmapMemObject(cq, s2m, s2, 0, NULL, NULL);
   check(err, "clEnqueueUnmapMemObject s2");

   clFinish(cq);

   printf("Setting kernel arguments...\n");
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &s1m);
   check(err, "clSetKernelArg 0");
   err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &s2m);
   check(err, "clSetKernelArg 1");
   err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dm);
   check(err, "clSetKernelArg 2");

   printf("Enqueueing NDRangeKernel...\n");
   size_t local[3] = {16, 1, 1};
   size_t global[3] = {1024, 1, 1};
   err = clEnqueueNDRangeKernel(cq, kernel, 3, NULL, global, local, 0, NULL, &eventK);
   check(err, "clEnqueueNDRangeKernel");

   printf("Enqueueing ReadBuffer...\n");
   err = clEnqueueReadBuffer(cq, dm, CL_FALSE, 0, REALSIZE, d, 0, NULL, &eventR);
   check(err, "clEnqueueReadBuffer");

   printf("Finishing queue...\n");
   clFinish(cq);

   printf("Data...\n");
   {
      int j;
      for (j=0; j<SIZE; j++) {
        printf("%f ", d[j]);
      }
      printf("\n");
   }

#ifdef PROFILING
   #define DURATION(event,label) do { \
      cl_ulong t0,t1; \
      err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, NULL);\
      check(err, "clGetEventProfilingInfo");\
      err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t1, NULL);\
      check(err, "clGetEventProfilingInfo");\
      printf("Profiling %s: %lu nanoseconds\n", label, t1-t0);\
   } while (0);

   DURATION(eventK, "kernel execution");
   DURATION(eventR, "result buffer reading");
#endif


   printf("Releasing events...\n");
   err = clReleaseEvent(eventK);
   err |= clReleaseEvent(eventR);
   check(err, "clReleaseCommandQueue");

   printf("Releasing command queue...\n");
   err = clReleaseCommandQueue(cq);
   check(err, "clReleaseCommandQueue");

   printf("Releasing buffers...\n");
   err = clReleaseMemObject(s1m);
   check(err, "clReleaseMemObject s1");
   err = clReleaseMemObject(s2m);
   check(err, "clReleaseMemObject s2");
   err = clReleaseMemObject(dm);
   check(err, "clReleaseMemObject d");

   printf("Releasing kernel...\n");
   err = clReleaseKernel(kernel);
   check(err, "clReleaseKernel");

   printf("Releasing program...\n");
   err = clReleaseProgram(program);
   check(err, "clReleaseProgram");

   printf("Releasing context...\n");
   err = clReleaseContext(context);
   check(err, "clReleaseContext");

#ifdef HAVE_CLGETEXTENSIONFUNCTIONADDRESSFORPLATFORM
   void (*clShutdown)(void) = clGetExtensionFunctionAddressForPlatform(platforms[platform_idx], "clShutdown");

   if (clShutdown != NULL) {
	   printf("Calling clShutdown :)\n");
	   clShutdown();
   }
#endif

   return 0;
}
