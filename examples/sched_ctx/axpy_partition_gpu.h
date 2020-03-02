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

#pragma once


__device__ static uint get_smid(void)
{
#if defined(__CUDACC__)
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
#else
  return 0;
#endif
}


#define __P_HKARGS    dimGrid,     active_blocks     ,occupancy,               block_assignment_d,   mapping_start
#define __P_KARGS dim3 blocks, int active_blocks, int occupancy, unsigned int* block_assignment, int mapping_start

#define __P_DARGS blocks,blockid

#define __P_BEGIN							\
__shared__ unsigned int block_start;					\
int smid = get_smid();							\
if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)		\
  {									\
    block_start = atomicDec(&block_assignment[smid],0xDEADBEEF);	\
  }									\
__syncthreads();							\
									\
if(block_start > active_blocks)						\
  {									\
    return;								\
  }

#define __P_LOOPXY							\
  dim3 blockid;								\
  blockid.z = 0;							\
									\
  int gridDim_sum = blocks.x*blocks.y;					\
  int startBlock = block_start + (smid - mapping_start) * occupancy;	\
									\
  for(int blockid_sum = startBlock; blockid_sum < gridDim_sum; blockid_sum +=active_blocks) \
    {									\
  blockid.x = blockid_sum % blocks.x;					\
  blockid.y = blockid_sum / blocks.x;

#define __P_LOOPEND }
// Needed if shared memory is used
#define __P_LOOPEND_SAFE __syncthreads(); }

#define __P_LOOPX							\
  dim3 blockid;								\
  blockid.z = 0;							\
  blockid.y = 0;							\
  int gridDim_sum = blocks.x;						\
  int startBlock = (smid-mapping_start) + block_start*(active_blocks/occupancy); \
									\
  for(int blockid_sum = startBlock; blockid_sum < gridDim_sum; blockid_sum +=active_blocks) \
    {									\
  blockid.x = blockid_sum;


  //  int startBlock = block_start + (smid - mapping_start) * occupancy; \


//////////// HOST side functions


template <typename F>
static void buildPartitionedBlockMapping(F cudaFun, int threads, int shmem, int mapping_start, int allocation,
				  int &width, int &active_blocks, unsigned int *block_assignment_d,cudaStream_t current_stream =
#ifdef cudaStreamPerThread
				  cudaStreamPerThread
#else
				  NULL
#endif
				  )
{
  int occupancy;
  int nb_SM = 13; //TODO: replace with call
  int mapping_end = mapping_start + allocation - 1; // exclusive
  unsigned int block_assignment[15];
  
#if CUDART_VERSION >= 6050
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,cudaFun,threads,shmem);
#else
  occupancy = 4;
#endif
  width = occupancy * nb_SM; // Physical wrapper grid size. Fits GPU exactly
  active_blocks = occupancy*allocation; // The total number of blocks doing work

  for(int i = 0; i < mapping_start; i++)
    block_assignment[i] = (unsigned) -1;

  for(int i = mapping_start; i <= mapping_end; i++)
    {
      block_assignment[i] = occupancy - 1;
    }

  for(int i = mapping_end+1; i < nb_SM; i++)
    block_assignment[i] = (unsigned) -1;

  cudaMemcpyAsync((void*)block_assignment_d,block_assignment,sizeof(block_assignment),cudaMemcpyHostToDevice, current_stream);
  //cudaMemcpy((void*)block_assignment_d,block_assignment,sizeof(block_assignment),cudaMemcpyHostToDevice);
  //cudaDeviceSynchronize();
}



#define __P_HOSTSETUP(KERNEL,GRIDDIM,BLOCKSIZE,SHMEMSIZE,MAPPING_START,MAPPING_END,STREAM)	\
  unsigned int* block_assignment_d; cudaMalloc((void**) &block_assignment_d,15*sizeof(unsigned int)); \
  int width = 0;							\
  int active_blocks = 0;						\
  buildPartitionedBlockMapping(KERNEL,BLOCKSIZE,SHMEMSIZE,(MAPPING_START),(MAPPING_END)-(MAPPING_START), \
			       width, active_blocks, block_assignment_d,STREAM); \
  int occupancy = active_blocks/((MAPPING_END)-(MAPPING_START));		\
  dim3 dimGrid = (GRIDDIM);\
  int mapping_start = (MAPPING_START);
