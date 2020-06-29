/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Mael Keryell
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
#include <stdint.h>
#include <starpu.h>

#define THREADS_PER_BLOCK 64

static inline long long jlstarpu_max(long long a, long long b)
{
	return (a > b) ? a : b;
}

static inline long long jlstarpu_interval_size(long long start, long long step, long long stop)
{
    if (stop >= start){
            return jlstarpu_max(0, (stop - start + 1) / step);
    } else {
            return jlstarpu_max(0, (stop - start - 1) / step);
    }
}


__device__ static inline long long jlstarpu_max__device(long long a, long long b)
{
	return (a > b) ? a : b;
}

__device__ static inline long long jlstarpu_interval_size__device(long long start, long long step, long long stop)
{
	if (stop >= start){
		return jlstarpu_max__device(0, (stop - start + 1) / step);
	} else {
		return jlstarpu_max__device(0, (stop - start - 1) / step);
	}
}


__global__ void mandelbrot(int64_t kernel_ids__start_1, int64_t kernel_ids__step_1, int64_t kernel_ids__dim_1, int64_t kernel_ids__start_2, 
                           int64_t kernel_ids__step_2, int64_t kernel_ids__dim_2, double* ptr_hF6lCYyJ, int64_t local_width, 
                           int64_t* ptr_qoUGBRtY, int64_t local_height, double conv_limit, int64_t* ptr_A5zD9sJZ, 
                           uint32_t ld_A5zD9sJZ)
{
    int64_t THREAD_ID = (int64_t) ((((blockIdx).x) * ((blockDim).x)) + ((threadIdx).x));
    
    if ((THREAD_ID) >= (((1) * (kernel_ids__dim_2)) * (kernel_ids__dim_1)))
    {
        return ;
    };
    int64_t kernel_ids__index_1 = (int64_t) (((THREAD_ID) / ((1) * (kernel_ids__dim_2))) % (kernel_ids__dim_1));
    int64_t kernel_ids__index_2 = (int64_t) (((THREAD_ID) / (1)) % (kernel_ids__dim_2));
    int64_t x = (int64_t) ((kernel_ids__start_1) + ((kernel_ids__index_1) * (kernel_ids__step_1)));
    int64_t y = (int64_t) ((kernel_ids__start_2) + ((kernel_ids__index_2) * (kernel_ids__step_2)));
    double max_iterations = (double) (ptr_hF6lCYyJ[(5) - (1)]);
    double zoom = (double) ((ptr_hF6lCYyJ[(3) - (1)]) * (0.25296875));
    int64_t X = (int64_t) ((x) + ((local_width) * ((ptr_qoUGBRtY[(2) - (1)]) - (1))));
    int64_t Y = (int64_t) ((y) + ((local_height) * ((ptr_qoUGBRtY[(1) - (1)]) - (1))));
    double cr = (double) ((ptr_hF6lCYyJ[(1) - (1)]) + (((X) - ((ptr_hF6lCYyJ[(3) - (1)]) / (2))) / (zoom)));
    double zr = (double) (cr);
    double ci = (double) ((ptr_hF6lCYyJ[(2) - (1)]) + (((Y) - ((ptr_hF6lCYyJ[(4) - (1)]) / (2))) / (zoom)));
    double zi = (double) (ci);
    int64_t n = (int64_t) (0);
    int64_t b1 = (int64_t) (((n) < (max_iterations)) + ((((zr) * (zr)) + ((zi) * (zi))) < ((conv_limit) * (conv_limit))));
    
    while ((b1) >= (2))
    {
        double tmp = (double) ((((zr) * (zr)) - ((zi) * (zi))) + (cr));
        zi = ((2) * (zr) * (zi)) + (ci);
        zr = tmp;
        n = (n) + (1);
        b1 = ((n) <= (max_iterations)) + ((((zr) * (zr)) + ((zi) * (zi))) <= ((conv_limit) * (conv_limit)));
    }
    ;
    
    if ((n) < (max_iterations))
    {
        ptr_A5zD9sJZ[((y) + (((x) - (1)) * (ld_A5zD9sJZ))) - (1)] = ((255) * (n)) / (max_iterations);
    } else
    {
        ptr_A5zD9sJZ[((y) + (((x) - (1)) * (ld_A5zD9sJZ))) - (1)] = 0;
    }
    ;
}



extern "C" void CUDA_mandelbrot(void** buffers_uwrYFDVe, void* cl_arg_uwrYFDVe)
{
    uint32_t ld_A5zD9sJZ = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_uwrYFDVe[(1) - (1)]));
    int64_t* ptr_A5zD9sJZ = (int64_t*) (STARPU_MATRIX_GET_PTR(buffers_uwrYFDVe[(1) - (1)]));
    double* ptr_hF6lCYyJ = (double*) (STARPU_VECTOR_GET_PTR(buffers_uwrYFDVe[(2) - (1)]));
    int64_t* ptr_qoUGBRtY = (int64_t*) (STARPU_VECTOR_GET_PTR(buffers_uwrYFDVe[(3) - (1)]));
    int64_t local_width = (int64_t) (STARPU_MATRIX_GET_NY(buffers_uwrYFDVe[(1) - (1)]));
    int64_t local_height = (int64_t) (STARPU_MATRIX_GET_NX(buffers_uwrYFDVe[(1) - (1)]));
    double conv_limit = (double) (2.0);
    int64_t kernel_ids__start_1 = (int64_t) (1);
    int64_t kernel_ids__step_1 = (int64_t) (1);
    int64_t kernel_ids__dim_1 = (int64_t) (jlstarpu_interval_size(kernel_ids__start_1, kernel_ids__step_1, local_width));
    int64_t kernel_ids__start_2 = (int64_t) (1);
    int64_t kernel_ids__step_2 = (int64_t) (1);
    int64_t kernel_ids__dim_2 = (int64_t) (jlstarpu_interval_size(kernel_ids__start_2, kernel_ids__step_2, local_height));
    int64_t nthreads = (int64_t) (((1) * (kernel_ids__dim_1)) * (kernel_ids__dim_2));
    int64_t nblocks = (int64_t) ((((nthreads) + (THREADS_PER_BLOCK)) - (1)) / (THREADS_PER_BLOCK));
    
    mandelbrot
        <<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
        >>> (kernel_ids__start_1, kernel_ids__step_1, kernel_ids__dim_1, kernel_ids__start_2, 
             kernel_ids__step_2, kernel_ids__dim_2, ptr_hF6lCYyJ, local_width, 
             ptr_qoUGBRtY, local_height, conv_limit, ptr_A5zD9sJZ, 
             ld_A5zD9sJZ);
    ;
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


