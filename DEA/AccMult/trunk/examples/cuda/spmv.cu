//#include "spmv_cuda.h"
#include <stdint.h>

#define MIN(a,b)	((a)<(b)?(a):(b))

extern "C" __global__ 
void spmv_kernel(uint32_t nnz, uint32_t nrow, uint32_t _nzval, uint32_t _colind, uint32_t _rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			uint32_t ptr_in, uint32_t nx_in, uint32_t ny_in, uint32_t ld_in,
			uint32_t ptr_out, uint32_t nx_out, uint32_t ny_out, uint32_t ld_out)
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

		unsigned firstindex = rowptr[row];
		unsigned lastindex = (row < nrow - 1)?rowptr[row+1]:nnz;

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
void spmv_kernel_2(uint32_t nnz, uint32_t nrow, uint32_t _nzval, uint32_t _colind, uint32_t _rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			uint32_t ptr_in, uint32_t nx_in, uint32_t ny_in, uint32_t ld_in,
			uint32_t ptr_out, uint32_t nx_out, uint32_t ny_out, uint32_t ld_out)
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

		unsigned firstindex = rowptr[row];
		unsigned lastindex = (row < nrow - 1)?rowptr[row+1]:nnz;

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
void spmv_kernel_3(uint32_t nnz, uint32_t nrow, uint32_t _nzval, uint32_t _colind, uint32_t _rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			uint32_t ptr_in, uint32_t nx_in, uint32_t ny_in, uint32_t ld_in,
			uint32_t ptr_out, uint32_t nx_out, uint32_t ny_out, uint32_t ld_out)
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

		unsigned firstindex = rowptr[row];
		unsigned lastindex = (row < nrow - 1)?rowptr[row+1]:nnz;

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecin[col];
		}

		vecout[row] = tmp;
	}
	

}
