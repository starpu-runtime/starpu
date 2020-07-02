/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Uppsala University
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

/*
 * This creates two dumb vectors, splits them into chunks, and for each pair of
 * chunk, run axpy on them.
 */

#include <starpu.h>
#include "axpy_partition_gpu.h"
#include <stdio.h>

//This code demonstrates how to transform a kernel to execute on a given set of GPU SMs.


// Original kernel
__global__ void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n)  y[i] = a*x[i] + y[i];
}




// Transformed kernel
__global__ void saxpy_partitioned(__P_KARGS, int n, float a, float *x, float *y)
{
  __P_BEGIN;
  __P_LOOPX;
        int i = blockid.x*blockDim.x + threadIdx.x; // note that blockIdx is replaced.
	if (i<n)  y[i] = a*x[i] + y[i];
  __P_LOOPEND;
}
      

extern "C" void cuda_axpy(void *descr[], void *_args)
{
	 float a = *((float *)_args);

        unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

        float *x = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
        float *y = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	int SM_mapping_start = -1;
	int SM_mapping_end = -1; 
  	int SM_allocation = -1;
  
	cudaStream_t stream = starpu_cuda_get_local_stream();
	int workerid = starpu_worker_get_id();
    	starpu_sched_ctx_get_sms_interval(workerid, &SM_mapping_start, &SM_mapping_end);
	SM_allocation = SM_mapping_end - SM_mapping_start;
	int dimensions = 512;	
	//partitioning setup
//	int SM_mapping_start = 0;
//  	int SM_allocation = 13;
  
	__P_HOSTSETUP(saxpy_partitioned,dim3(dimensions,1,1),dimensions,0,SM_mapping_start,SM_allocation,stream);

  	saxpy_partitioned<<<width,dimensions,0,stream>>>(__P_HKARGS,n,a,x,y);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
