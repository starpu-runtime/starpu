/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* This example shows a basic StarPU vector scale app on top of StarPURM with a nVidia CUDA kernel */

#include <starpu.h>
#include <starpurm.h>

static __global__ void vector_scale_cuda_kernel(float *vector, unsigned n, float scalar)
{
	unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		vector[i] *= scalar;
	}
}

extern "C" void vector_scale_cuda_func(void *cl_buffers[], void *cl_arg)
{
	float scalar = -1.0;
	unsigned n = STARPU_VECTOR_GET_NX(cl_buffers[0]);
	float *vector = (float *)STARPU_VECTOR_GET_PTR(cl_buffers[0]);
	starpu_codelet_unpack_args(cl_arg, &scalar);

	{
		int workerid = starpu_worker_get_id();
		hwloc_cpuset_t worker_cpuset = starpu_worker_get_hwloc_cpuset(workerid);
		hwloc_cpuset_t check_cpuset = starpurm_get_selected_cpuset();
#if 0
		{
			int strl1 = hwloc_bitmap_snprintf(NULL, 0, worker_cpuset);
			char str1[strl1+1];
			hwloc_bitmap_snprintf(str1, strl1+1, worker_cpuset);
			int strl2 = hwloc_bitmap_snprintf(NULL, 0, check_cpuset);
			char str2[strl2+1];
			hwloc_bitmap_snprintf(str2, strl2+1, check_cpuset);
			printf("worker[%03d] - task: vector=%p, n=%d, scalar=%lf, worker cpuset = %s, selected cpuset = %s\n", workerid, vector, n, scalar, str1, str2);
		}
#endif
		hwloc_bitmap_and(check_cpuset, check_cpuset, worker_cpuset);
		assert(!hwloc_bitmap_iszero(check_cpuset));
		hwloc_bitmap_free(check_cpuset);
		hwloc_bitmap_free(worker_cpuset);
	}

	unsigned nb_threads_per_block = 64;
	unsigned nb_blocks = (n + nb_threads_per_block-1) / nb_threads_per_block;
	vector_scale_cuda_kernel<<<nb_blocks,nb_threads_per_block,0,starpu_cuda_get_local_stream()>>>(vector, n, scalar);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
