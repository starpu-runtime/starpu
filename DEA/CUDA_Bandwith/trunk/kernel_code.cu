#include <stdint.h>
#include "param.h"

#define MIN(a,b) ((a)<(b)?(a):(b))


extern "C" __global__ 
void bandwith_test(int *src, int *dest, unsigned size , int *p)
{
//	unsigned i,j, actual_buffersize;
//	for (i = 0; i < size ; i += BUFFERSIZE) 
//	{
//		__syncthreads();
//		/* fill the buffer */
//		actual_buffersize = MIN(BUFFERSIZE, size - i);
//		for (j = 0; j < actual_buffersize; j++)
//		{
//			buffer[j] = src[i+j];
//		}
//
//		__syncthreads();
//		/* put those data back into the global memory */
//		for (j = 0; j < actual_buffersize; j++)
//		{
//			dest[i+j] = buffer[j];
//		}
//
//	}
//
	int i;
	for (i = 0; i < size; i++)
	{
		dest[i] = src[i];
	}

	*p = 42;

	return;
}
