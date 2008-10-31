#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

#define MIN(a,b)	((a)<(b)?(a):(b))

#include "spmv.cu"

//unsigned pinned = 0;
unsigned blocks = 1;
unsigned grids = 1;
uint32_t size = 1024*1024;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-nblocks") == 0) {
		        char *argptr;
			blocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-ngrids") == 0) {
		        char *argptr;
			grids = strtol(argv[++i], &argptr, 10);
		}



		if (strcmp(argv[i], "-size") == 0) {
		        char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

	}
}




int main(int argc, char** argv)
{
	CUT_DEVICE_INIT(argc, argv);

	parse_args(argc, argv);

	/* example of 3-band matrix */
	float *nzval;
	uint32_t nnz;
	uint32_t *colind;
	uint32_t *rowptr;

	float *d_nzval;
	uint32_t *d_colind;
	uint32_t *d_rowptr;

	nnz = 3*size-2;

	nzval = (float *)malloc(nnz*sizeof(float));
	colind = (uint32_t *)malloc(nnz*sizeof(uint32_t));
	rowptr = (uint32_t *)malloc((size+1)*sizeof(uint32_t));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_nzval, nnz*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_colind, nnz*sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_rowptr, (size+1)*sizeof(uint32_t)));

	assert(nzval);
	assert(colind);
	assert(rowptr);

	/* fill the matrix */
	unsigned row;
	unsigned pos = 0;
	for (row = 0; row < size; row++)
	{
		rowptr[row] = pos;

		if (row > 0) {
			nzval[pos] = 1.0f;
			colind[pos] = row-1;
			pos++;
		}
		
		nzval[pos] = 5.0f;
		colind[pos] = row;
		pos++;

		if (row < size - 1) {
			nzval[pos] = 1.0f;
			colind[pos] = row+1;
			pos++;
		}
	}

	rowptr[size] = nnz;

	/* initiate the 2 vectors */
	float *invec, *outvec;
	float *d_invec, *d_outvec;
	invec = (float *)malloc(size*sizeof(float));
	assert(invec);

	outvec = (float *)malloc(size*sizeof(float));
	assert(outvec);

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_invec, size*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_outvec, size*sizeof(float)));

	/* fill those */
	unsigned ind;
	for (ind = 0; ind < size; ind++)
	{
		invec[ind] = 2.0f;
		outvec[ind] = 0.0f;
	}

	/* upload the problem */
	CUDA_SAFE_CALL(cudaMemcpy((float *)d_nzval, nzval, nnz*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((float *)d_colind, colind, nnz*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((float *)d_rowptr, rowptr, (size+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy((float *)d_invec, invec, size*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((float *)d_outvec, outvec, size*sizeof(float), cudaMemcpyHostToDevice));

	/* launch the kernel */
	dim3 threads(blocks, 1);
	dim3 grid(grids, 1);

	spmv_kernel_3<<< grid, threads >>>(nnz, size, d_nzval, d_colind, d_rowptr, 0, sizeof(float), d_invec, size, d_outvec, size);
	cudaThreadSynchronize();

	/* download results */
	CUDA_SAFE_CALL(cudaMemcpy(outvec, (float *)d_outvec, size*sizeof(float), cudaMemcpyDeviceToHost) );

	printf("result \n");
	for (ind = 0; ind < MIN(size, 16); ind++)
	{
		printf("\t%2.2f\n", outvec[ind]);
	}
}
