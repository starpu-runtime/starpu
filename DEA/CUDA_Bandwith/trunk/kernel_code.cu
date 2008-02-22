#include <stdint.h>
#include "param.h"

#define UPDIV(a,b)	(((a) + (b) - 1) / (b))
#define MIN(a,b) ((a)<(b)?(a):(b))

__shared__ int buffer[BUFFERSIZE];

extern "C" __global__ 
#ifdef DEBUG
void bandwith_test_dumb(int *src, int *dest, unsigned size , int *p)
#else
void bandwith_test_dumb(int *src, int *dest, unsigned size)
#endif
{
	int i;
	for (i = 0; i < size; i++)
	{
		dest[i] = src[i];
	}

#ifdef DEBUG
	*p = 42;
#endif

	return;
}

extern "C" __global__ 
#ifdef DEBUG
void bandwith_test(int *src, int *dest, unsigned size , int *p)
#else
void bandwith_test(int *src, int *dest, unsigned size)
#endif
{
	unsigned i,j, actual_buffersize;
	for (i = 0; i < size ; i += BUFFERSIZE) 
	{
		__syncthreads();
		/* fill the buffer */
		actual_buffersize = MIN(BUFFERSIZE, size - i);
		for (j = 0; j < actual_buffersize; j++)
		{
			buffer[j] = src[i+j];
		}

		__syncthreads();
		/* put those data back into the global memory */
		for (j = 0; j < actual_buffersize; j++)
		{
			dest[i+j] = buffer[j];
		}

	}

#ifdef DEBUG
	*p = 42;
#endif

	return;
}


extern "C" __global__ 
#ifdef DEBUG
void bandwith_test_2(int *src, int *dest, unsigned size , int *p)
#else
void bandwith_test_2(int *src, int *dest, unsigned size)
#endif
{

	unsigned blockid = blockIdx.x + blockIdx.y*gridDim.x;
	unsigned threadid = threadIdx.x + threadIdx.y*blockDim.x;

	unsigned blockchunk_size = UPDIV(size, (gridDim.x * gridDim.y));

	unsigned blockchunk_start = MIN(blockchunk_size*blockid, size);
	unsigned blockchunk_end = MIN(blockchunk_size*(blockid+1), size);

	unsigned actual_blockchunk_size = blockchunk_end - blockchunk_start;


	unsigned threadchunk_size = UPDIV(actual_blockchunk_size, blockDim.x*blockDim.y);

	unsigned threadchunk_start = MIN(blockchunk_start + threadchunk_size*threadid, blockchunk_end);
	unsigned threadchunk_end = MIN(blockchunk_start + threadchunk_size*(threadid+1), blockchunk_end);

	unsigned i;
	for (i = threadchunk_start; i < threadchunk_end ; i++ ) 
	{
		dest[i] = src[i];
	}

#ifdef DEBUG
	*p = 42;
#endif

	return;
}

extern "C" __global__ 
#ifdef DEBUG
void bandwith_test_3(int *src, int *dest, unsigned size , int *p)
#else
void bandwith_test_3(int *src, int *dest, unsigned size)
#endif
{

	unsigned blockid = blockIdx.x + blockIdx.y*gridDim.x;
	unsigned threadid = threadIdx.x + threadIdx.y*blockDim.x;

	unsigned blockchunk_size = UPDIV(size, (gridDim.x * gridDim.y));

	unsigned blockchunk_start = MIN(blockchunk_size*blockid, size);
	unsigned blockchunk_end = MIN(blockchunk_size*(blockid+1), size);

	unsigned i;
	for (i = blockchunk_start + threadid; i < blockchunk_end ; i+=blockDim.x*blockDim.y ) 
	{
		dest[i] = src[i];
	}

#ifdef DEBUG
	*p = 42;
#endif

	return;
}
