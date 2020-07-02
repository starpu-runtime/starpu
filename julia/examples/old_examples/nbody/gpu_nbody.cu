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
#include <math.h>
#include <starpu.h>

struct Params
{
  unsigned taskx;
  unsigned epsilon;
};

__global__ void gpuNbodyKernel(double *P, double *subA, double *M,
			     uint32_t nxP, uint32_t nxA, uint32_t nxM,
			     uint32_t ldP, uint32_t ldA,
			     struct Params params)
{
  uint32_t id, i, j, k;
  double dx, dy, modul;

  id = blockIdx.x * blockDim.x + threadIdx.x;
  i = id % nxA;
  j = id / nxA;

  if (j >= 1){
    return;
  }

  double sumaccx;
  double sumaccy;
  
  for (k = 0; k < nxP; k++){
    if (k != id + nxA*params.taskx){
      dx = P[k] - P[id + nxA*params.taskx];
      dy = P[k + ldP] - P[id + nxA*params.taskx + ldP];
      
      modul = dx * dx + dy * dy;

      sumaccx = 6.674e-11 * M[k] * dx / pow(modul + params.epsilon, 3);
      sumaccy = 6.674e-11 * M[k] * dy / pow(modul + params.epsilon, 3);
    }
  }
 
  subA[i] = sumaccx;
  subA[i + ldA] = sumaccy;

  // P[id + nxA * params.taskx] = subA[i];

  // subA[i] = 0;
  // subA[i + ldA] = 1;
  
}

#define THREADS_PER_BLOCK 64

extern "C" void gpu_nbody(void * descr[], void * args)
{

  double *d_P, *d_subA, *d_M;
  uint32_t nxP, nxA, nxM;
  uint32_t ldA, ldP;
  uint32_t nblocks;

  struct Params *params = (struct Params *) args;

  d_P = (double *) STARPU_MATRIX_GET_PTR(descr[0]);
  d_subA = (double *) STARPU_MATRIX_GET_PTR(descr[1]);
  d_M = (double *) STARPU_MATRIX_GET_PTR(descr[2]);

  nxP = STARPU_MATRIX_GET_NX(descr[0]);
  nxA = STARPU_MATRIX_GET_NX(descr[1]);
  nxM = STARPU_MATRIX_GET_NX(descr[2]);

  ldP = STARPU_MATRIX_GET_LD(descr[0]);
  ldA = STARPU_MATRIX_GET_LD(descr[1]);

  nblocks = (nxA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  gpuNbodyKernel
    <<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
    >>> (d_P,  d_subA, d_M, nxP, nxA, nxM, ldP, ldA, *params);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());

}







__global__ void gpuNbody2Kernel(double *d_subP, double *d_subV, double *d_subA,
			      uint32_t nxP, uint32_t nxV, uint32_t nxA,
			      uint32_t ldP, uint32_t ldV, uint32_t ldA,
			      struct Params params)
{

  uint32_t id, i, j;

  id = blockIdx.x * blockDim.x + threadIdx.x;

  i = id % nxP;
  j = id / nxP;

  if (j >= 1){
    return;
  }

  d_subV[i] = d_subV[i] + 3600*d_subA[i];
  d_subV[i + ldV] = d_subV[i + ldV] + 3600*d_subA[i + ldA];

  d_subP[i] = d_subP[i] + 3600*d_subV[i];
  d_subP[i + ldP] = d_subP[i + ldP] + 3600*d_subV[i + ldV];
}


extern "C" void gpu_nbody2(void * descr[], void *args)
{
  double *d_subP, *d_subV, *d_subA;
  uint32_t nxP, nxV, nxA;
  uint32_t ldP, ldV, ldA;
  uint32_t nblocks;

  struct Params *params = (struct Params *) args;

  d_subP = (double *) STARPU_MATRIX_GET_PTR(descr[0]);
  d_subV = (double *) STARPU_MATRIX_GET_PTR(descr[1]);
  d_subA = (double *) STARPU_MATRIX_GET_PTR(descr[2]);

  nxP = STARPU_MATRIX_GET_NX(descr[0]);
  nxV = STARPU_MATRIX_GET_NX(descr[1]);
  nxA = STARPU_MATRIX_GET_NX(descr[2]);

  ldP = STARPU_MATRIX_GET_LD(descr[0]);
  ldV = STARPU_MATRIX_GET_LD(descr[1]);
  ldA = STARPU_MATRIX_GET_LD(descr[2]);

  nblocks = (nxA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  gpuNbody2Kernel
    <<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
    >>> (d_subP, d_subV, d_subA, nxP, nxV, nxA, ldP, ldV, ldA, *params);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
