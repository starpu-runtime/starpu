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
#include <starpu.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

struct Params
{
  float cr;
  float ci;
  unsigned taskx;
  unsigned tasky;
  unsigned width;
  unsigned height;
};


__global__ void gpuMandelbrotKernel
(
 uint32_t nxP, uint32_t nyP, 
 uint32_t ldP,
 int * subP,
 struct Params params
 )
{
  unsigned width = params.width;
  unsigned height = params.height;
  unsigned taskx = params.taskx;
  unsigned tasky = params.tasky;

  float centerr = params.cr;
  float centeri = params.ci;

  float zoom = width * 0.25296875;
  int maxiter = (width/2) * 0.049715909 * log10(zoom);
  float conv_lim = 2.0;

  uint32_t id;
  int n,i,j,x,y;
  float zr,zi,cr,ci;

  id = blockIdx.x * blockDim.x + threadIdx.x;
  i = id % nxP;
  j = id / nxP;
  

  if (j >= nyP){
    return;
  }

  x = i + taskx * nxP;
  y = j + tasky * nyP;
  
  zr = cr = centerr + (x - width/2.)/zoom;
  zi = ci = centeri + (y - height/2.)/zoom;
  float m = zr*zr + zi*zi;
  
  for (n = 0; n <= maxiter && m < conv_lim * conv_lim; n++) {
    float tmp = zr * zr - zi*zi + cr;
    zi = 2*zr*zi + ci;
    zr = tmp;
    m = zr*zr + zi*zi;
  }
  int color;
  if (n < maxiter)
     color = 255.*n/maxiter;
   else
     color = 0;
  subP[i + j*ldP] = color;

}


#define THREADS_PER_BLOCK 64

extern "C" void gpu_mandelbrot(void *descr[], void *args)
{
  int *d_subP;
  uint32_t nxP, nyP;
  uint32_t ldP;
  uint32_t nblocks;
  struct Params *params = (struct Params *) args;

  d_subP = (int *) STARPU_MATRIX_GET_PTR(descr[0]);

  nxP = STARPU_MATRIX_GET_NX(descr[0]);
  nyP = STARPU_MATRIX_GET_NY(descr[0]);

  ldP = STARPU_MATRIX_GET_LD(descr[0]);

  nblocks = (nxP * nyP + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

  gpuMandelbrotKernel <<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream() >>> (nxP, nyP, ldP, d_subP, *params);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
