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


__global__ void nbody_acc(int64_t kernel_ids__start_1, int64_t kernel_ids__step_1, int64_t kernel_ids__dim_1, int64_t widthp, 
                          double* ptr_OhLp7E87, int64_t* ptr_DvYAWLG1, int64_t widtha, double* ptr_Xi5IjQJ9, 
                          uint32_t ld_Xi5IjQJ9, double* ptr_t4YHT0eY, double* ptr_mfUSUHkf, uint32_t ld_mfUSUHkf)
{
    int64_t THREAD_ID = (int64_t) ((((blockIdx).x) * ((blockDim).x)) + ((threadIdx).x));
    
    if ((THREAD_ID) >= ((1) * (kernel_ids__dim_1)))
    {
        return ;
    };
    int64_t kernel_ids__index_1 = (int64_t) (((THREAD_ID) / (1)) % (kernel_ids__dim_1));
    int64_t plan = (int64_t) ((kernel_ids__start_1) + ((kernel_ids__index_1) * (kernel_ids__step_1)));
    double sumaccx = (double) (0);
    double sumaccy = (double) (0);
    
    int64_t start_TzfU6QY7 = (int64_t) (1);
    int64_t stop_TzfU6QY7 = (int64_t) (widthp);
    int64_t oplan;

    for (oplan = start_TzfU6QY7 ; oplan <= stop_TzfU6QY7 ; oplan += 1)
    {
        double eps = (double) (ptr_OhLp7E87[(3) - (1)]);
        int64_t Id = (int64_t) ((ptr_DvYAWLG1[(1) - (1)]) * (widtha));
        double G = (double) (ptr_OhLp7E87[(1) - (1)]);
        int64_t b = (int64_t) ((((plan) + (Id)) >= (oplan)) + (((plan) + (Id)) <= (oplan)));
        
        if ((b) < (2))
        {
            double dx = (double) ((ptr_Xi5IjQJ9[((1) + (((oplan) - (1)) * (ld_Xi5IjQJ9))) - (1)]) - (ptr_Xi5IjQJ9[((1) + ((((plan) + (Id)) - (1)) * (ld_Xi5IjQJ9))) - (1)]));
            double dy = (double) ((ptr_Xi5IjQJ9[((2) + (((oplan) - (1)) * (ld_Xi5IjQJ9))) - (1)]) - (ptr_Xi5IjQJ9[((2) + ((((plan) + (Id)) - (1)) * (ld_Xi5IjQJ9))) - (1)]));
            double modul = (double) (sqrt(((dx) * (dx)) + ((dy) * (dy))));
            sumaccx = (sumaccx) + (((G) * (ptr_t4YHT0eY[(oplan) - (1)]) * (dx)) / (((modul) + (eps)) * ((modul) + (eps)) * ((modul) + (eps))));
            sumaccy = (sumaccy) + (((G) * (ptr_t4YHT0eY[(oplan) - (1)]) * (dy)) / (((modul) + (eps)) * ((modul) + (eps)) * ((modul) + (eps))));
        };
    }
    ;
    ptr_mfUSUHkf[((1) + (((plan) - (1)) * (ld_mfUSUHkf))) - (1)] = sumaccx;
    ptr_mfUSUHkf[((2) + (((plan) - (1)) * (ld_mfUSUHkf))) - (1)] = sumaccy;
}



extern "C" void CUDA_nbody_acc(void** buffers_qd9i9yfK, void* cl_arg_qd9i9yfK)
{
    uint32_t ld_Xi5IjQJ9 = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_qd9i9yfK[(1) - (1)]));
    double* ptr_Xi5IjQJ9 = (double*) (STARPU_MATRIX_GET_PTR(buffers_qd9i9yfK[(1) - (1)]));
    uint32_t ld_mfUSUHkf = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_qd9i9yfK[(2) - (1)]));
    double* ptr_mfUSUHkf = (double*) (STARPU_MATRIX_GET_PTR(buffers_qd9i9yfK[(2) - (1)]));
    double* ptr_t4YHT0eY = (double*) (STARPU_VECTOR_GET_PTR(buffers_qd9i9yfK[(3) - (1)]));
    double* ptr_OhLp7E87 = (double*) (STARPU_VECTOR_GET_PTR(buffers_qd9i9yfK[(4) - (1)]));
    int64_t* ptr_DvYAWLG1 = (int64_t*) (STARPU_VECTOR_GET_PTR(buffers_qd9i9yfK[(5) - (1)]));
    int64_t widthp = (int64_t) (STARPU_MATRIX_GET_NY(buffers_qd9i9yfK[(1) - (1)]));
    int64_t widtha = (int64_t) (STARPU_MATRIX_GET_NY(buffers_qd9i9yfK[(2) - (1)]));
    int64_t kernel_ids__start_1 = (int64_t) (1);
    int64_t kernel_ids__step_1 = (int64_t) (1);
    int64_t kernel_ids__dim_1 = (int64_t) (jlstarpu_interval_size(kernel_ids__start_1, kernel_ids__step_1, widtha));
    int64_t nthreads = (int64_t) ((1) * (kernel_ids__dim_1));
    int64_t nblocks = (int64_t) ((((nthreads) + (THREADS_PER_BLOCK)) - (1)) / (THREADS_PER_BLOCK));
    
    nbody_acc
        <<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
        >>> (kernel_ids__start_1, kernel_ids__step_1, kernel_ids__dim_1, widthp, 
             ptr_OhLp7E87, ptr_DvYAWLG1, widtha, ptr_Xi5IjQJ9, 
             ld_Xi5IjQJ9, ptr_t4YHT0eY, ptr_mfUSUHkf, ld_mfUSUHkf);
    ;
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


__global__ void nbody_updt(int64_t kernel_ids__start_1, int64_t kernel_ids__step_1, int64_t kernel_ids__dim_1, double* ptr_jJ5f8wMA, 
                           uint32_t ld_jJ5f8wMA, double* ptr_piPvdbTs, uint32_t ld_piPvdbTs, double* ptr_JBaPgPiT, 
                           double* ptr_0STm2S4k, uint32_t ld_0STm2S4k)
{
    int64_t THREAD_ID = (int64_t) ((((blockIdx).x) * ((blockDim).x)) + ((threadIdx).x));
    
    if ((THREAD_ID) >= ((1) * (kernel_ids__dim_1)))
    {
        return ;
    };
    int64_t kernel_ids__index_1 = (int64_t) (((THREAD_ID) / (1)) % (kernel_ids__dim_1));
    int64_t i = (int64_t) ((kernel_ids__start_1) + ((kernel_ids__index_1) * (kernel_ids__step_1)));
    ptr_jJ5f8wMA[((1) + (((i) - (1)) * (ld_jJ5f8wMA))) - (1)] = (ptr_jJ5f8wMA[((1) + (((i) - (1)) * (ld_jJ5f8wMA))) - (1)]) + ((ptr_piPvdbTs[((1) + (((i) - (1)) * (ld_piPvdbTs))) - (1)]) * (ptr_JBaPgPiT[(2) - (1)]));
    ptr_jJ5f8wMA[((2) + (((i) - (1)) * (ld_jJ5f8wMA))) - (1)] = (ptr_jJ5f8wMA[((2) + (((i) - (1)) * (ld_jJ5f8wMA))) - (1)]) + ((ptr_piPvdbTs[((2) + (((i) - (1)) * (ld_piPvdbTs))) - (1)]) * (ptr_JBaPgPiT[(2) - (1)]));
    ptr_0STm2S4k[((1) + (((i) - (1)) * (ld_0STm2S4k))) - (1)] = (ptr_0STm2S4k[((1) + (((i) - (1)) * (ld_0STm2S4k))) - (1)]) + ((ptr_jJ5f8wMA[((1) + (((i) - (1)) * (ld_jJ5f8wMA))) - (1)]) * (ptr_JBaPgPiT[(2) - (1)]));
    ptr_0STm2S4k[((2) + (((i) - (1)) * (ld_0STm2S4k))) - (1)] = (ptr_0STm2S4k[((2) + (((i) - (1)) * (ld_0STm2S4k))) - (1)]) + ((ptr_jJ5f8wMA[((2) + (((i) - (1)) * (ld_jJ5f8wMA))) - (1)]) * (ptr_JBaPgPiT[(2) - (1)]));
}



extern "C" void CUDA_nbody_updt(void** buffers_gj6UYWT4, void* cl_arg_gj6UYWT4)
{
    uint32_t ld_0STm2S4k = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_gj6UYWT4[(1) - (1)]));
    double* ptr_0STm2S4k = (double*) (STARPU_MATRIX_GET_PTR(buffers_gj6UYWT4[(1) - (1)]));
    uint32_t ld_jJ5f8wMA = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_gj6UYWT4[(2) - (1)]));
    double* ptr_jJ5f8wMA = (double*) (STARPU_MATRIX_GET_PTR(buffers_gj6UYWT4[(2) - (1)]));
    uint32_t ld_piPvdbTs = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_gj6UYWT4[(3) - (1)]));
    double* ptr_piPvdbTs = (double*) (STARPU_MATRIX_GET_PTR(buffers_gj6UYWT4[(3) - (1)]));
    double* ptr_JBaPgPiT = (double*) (STARPU_VECTOR_GET_PTR(buffers_gj6UYWT4[(4) - (1)]));
    int64_t widthp = (int64_t) (STARPU_MATRIX_GET_NY(buffers_gj6UYWT4[(1) - (1)]));
    int64_t kernel_ids__start_1 = (int64_t) (1);
    int64_t kernel_ids__step_1 = (int64_t) (1);
    int64_t kernel_ids__dim_1 = (int64_t) (jlstarpu_interval_size(kernel_ids__start_1, kernel_ids__step_1, widthp));
    int64_t nthreads = (int64_t) ((1) * (kernel_ids__dim_1));
    int64_t nblocks = (int64_t) ((((nthreads) + (THREADS_PER_BLOCK)) - (1)) / (THREADS_PER_BLOCK));
    
    nbody_updt
        <<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
        >>> (kernel_ids__start_1, kernel_ids__step_1, kernel_ids__dim_1, ptr_jJ5f8wMA, 
             ld_jJ5f8wMA, ptr_piPvdbTs, ld_piPvdbTs, ptr_JBaPgPiT, 
             ptr_0STm2S4k, ld_0STm2S4k);
    ;
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


