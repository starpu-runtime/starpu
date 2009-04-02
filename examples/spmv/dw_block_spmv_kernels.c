/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "dw_block_spmv.h"

/*
 *   U22 
 */

static inline void common_block_spmv(starpu_data_interface_t *buffers, int s, __attribute__((unused)) void *_args)
{
	//printf("22\n");
	float *block 	= (float *)buffers[0].blas.ptr;
	float *in 	= (float *)buffers[1].vector.ptr;
	float *out 	= (float *)buffers[2].vector.ptr;

	unsigned dx = buffers[0].blas.nx;
	unsigned dy = buffers[0].blas.ny;

	unsigned ld = buffers[0].blas.ld;

	switch (s) {
		case 0:
			cblas_sgemv(CblasRowMajor, CblasNoTrans, dx, dy, 1.0f, block, ld, in, 1, 1.0f, out, 1);
			break;
#ifdef USE_CUDA
		case 1:
			cublasSgemv ('t', dx, dy, 1.0f, block, ld, in, 1, 1.0f, out, 1);
			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void core_block_spmv(starpu_data_interface_t *descr, void *_args)
{
//	printf("CORE CODELET \n");

	common_block_spmv(descr, 0, _args);
}

#ifdef USE_CUDA
void cublas_block_spmv(starpu_data_interface_t *descr, void *_args)
{
//	printf("CUBLAS CODELET \n");

	common_block_spmv(descr, 1, _args);
}
#endif// USE_CUDA
