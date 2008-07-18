#include <stdint.h>

#define MIN(a,b)	((a)<(b)?(a):(b))

extern "C" __global__ 
void spmv_kernel(uint32_t nnz, uint32_t nrow, uintptr_t _nzval, uint32_t *_colind, uint32_t *_rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			uintptr_t ptr_in, uint32_t nx_in, uintptr_t ptr_out, uint32_t nx_out)
{
	float *vecin = (float *)ptr_in;
	float *vecout = (float *)ptr_out;
	float *nzval = (float *)_nzval;

	uint32_t *rowptr = (uint32_t *)_rowptr;
	uint32_t *colind = (uint32_t *)_colind;

	/* only one dimension is used here */
	unsigned nthreads = gridDim.x*blockDim.x;
	unsigned threadid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned rowstart = threadid * ((nrow + (nthreads - 1))/nthreads);
	unsigned rowend = MIN(nrow, (threadid+1) * ((nrow + (nthreads - 1))/nthreads));

	unsigned row;
	for (row = rowstart; row < rowend; row++)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstentry;
		unsigned lastindex = rowptr[row+1] - firstentry; 

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecin[col];
		}

		vecout[row] = tmp;
	}
}

extern "C" __global__ 
void spmv_kernel_2(uint32_t nnz, uint32_t nrow, uintptr_t _nzval, uint32_t *_colind, uint32_t *_rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			uintptr_t ptr_in, uint32_t nx_in, uintptr_t ptr_out, uint32_t nx_out)
{
	float *vecin = (float *)ptr_in;
	float *vecout = (float *)ptr_out;
	float *nzval = (float *)_nzval;

	uint32_t *rowptr = (uint32_t *)_rowptr;
	uint32_t *colind = (uint32_t *)_colind;

	/* only one dimension is used here */
	unsigned block_rowstart = blockIdx.x*( (nrow + gridDim.x - 1)/gridDim.x );
	unsigned block_rowend = MIN((blockIdx.x+1)*( (nrow + gridDim.x - 1)/gridDim.x ), nrow);

	unsigned row;
	for (row = block_rowstart + threadIdx.x; row < block_rowend; row+=blockDim.x)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstentry;
		unsigned lastindex = rowptr[row+1] - firstentry;

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecin[col];
		}

		vecout[row] = tmp;
	}
	

}



extern "C" __global__ 
void spmv_kernel_3(uint32_t nnz, uint32_t nrow, uintptr_t _nzval, uint32_t *_colind, uint32_t *_rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			uintptr_t ptr_in, uint32_t nx_in, uintptr_t ptr_out, uint32_t nx_out)
{
	float *vecin = (float *)ptr_in;
	float *vecout = (float *)ptr_out;
	float *nzval = (float *)_nzval;

	uint32_t *rowptr = (uint32_t *)_rowptr;
	uint32_t *colind = (uint32_t *)_colind;

	/* only one dimension is used here */
	unsigned block_rowstart = blockIdx.x*( (nrow + gridDim.x - 1)/gridDim.x );
	unsigned block_rowend = MIN((blockIdx.x+1)*( (nrow + gridDim.x - 1)/gridDim.x ), nrow);

	unsigned row;
	for (row = block_rowstart + threadIdx.x; row < block_rowend; row+=blockDim.x)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstentry;
		unsigned lastindex = rowptr[row+1] - firstentry;

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecin[col];
		}

		vecout[row] = tmp;
	}
	

}

#define ROWBUFSIZE	128
#define BUFSIZE		1024

extern "C" __global__ 
void spmv_kernel_3_test(uint32_t nnz, uint32_t nrow, float *nzval, uint32_t *colind, uint32_t *rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			float *vecin, uint32_t nx_in, float *vecout, uint32_t nx_out)
{

	/* only one dimension is used here */
	unsigned block_rowstart = blockIdx.x*( (nrow + gridDim.x - 1)/gridDim.x );
	unsigned block_rowend = MIN((blockIdx.x+1)*( (nrow + gridDim.x - 1)/gridDim.x ), nrow);

	unsigned processed = block_rowstart;


	//__shared__ uint32_t local_rowptr[ROWBUFSIZE];
	__shared__ float local_nzval[BUFSIZE];
	__shared__ uint32_t local_colind[BUFSIZE];
	
	
	while (processed < block_rowend) {
		/* find the size of the next block */
		unsigned firstblockindex = rowptr[processed] - firstentry;
		unsigned nextrow = processed;

		unsigned nextindex = firstblockindex;

		/* determine the size of the current block */
		while ((nextindex - firstblockindex < BUFSIZE) && nextrow < block_rowend) {
			nextrow++;
			nextindex = (rowptr[nextrow] - firstentry) - firstblockindex;
		}

		unsigned lastindex = rowptr[nextrow] - firstentry;
	
		/* fetch data into the shared data */
		unsigned ind;
		for (ind = firstblockindex + threadIdx.x; ind < nextindex; ind += blockDim.x)
		{
			local_nzval[ind - firstblockindex] = nzval[ind];
			local_colind[ind - firstblockindex] = colind[ind];
		}

		unsigned row;
		for (row = processed + threadIdx.x; row < nextrow; row+=blockDim.x)
		{
			float tmp = 0.0f;
			unsigned index;
	
			unsigned firstindex = rowptr[row] - firstblockindex;
			unsigned lastindex = rowptr[row+1] - firstblockindex;
	
			for (index = firstindex; index < lastindex; index++)
			{
				unsigned col;
	
				col = local_colind[index];
				tmp += local_nzval[index]*vecin[col];
			}

			vecout[row] = tmp;
		}

		processed = nextrow;
	}

}
