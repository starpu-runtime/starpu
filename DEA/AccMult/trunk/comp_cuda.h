#ifndef __COMP_CUDA_H__
#define __COMP_CUDA_H__

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <cuda.h>

#define UPDIV(a,b)	(((a)+(b)-1)/((b)))
//#define MIN(a,b)	((a)>(b)?(b):(a))

__device__ void cuda_dummy_mult(CUdeviceptr, CUdeviceptr, CUdeviceptr);

#endif // __COMP_CUDA_H__
