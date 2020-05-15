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
#include <stdint.h>
#include <starpu.h>

/*
 * The codelet is passed 3 matrices, the "descr" union-type field gives a
 * description of the layout of those 3 matrices in the local memory (ie. RAM
 * in the case of CPU, GPU frame buffer in the case of GPU etc.). Since we have
 * registered data with the "matrix" data interface, we use the matrix macros.
 */

void cpu_mult(void *descr[], void *arg)
{
	(void)arg;
	float *subA, *subB, *subC;
	uint32_t nxC, nyC, nyA;
	uint32_t ldA, ldB, ldC;

	/* .blas.ptr gives a pointer to the first element of the local copy */
	subA = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	subB = (float *)STARPU_MATRIX_GET_PTR(descr[1]);
	subC = (float *)STARPU_MATRIX_GET_PTR(descr[2]);


	/* .blas.nx is the number of rows (consecutive elements) and .blas.ny
	 * is the number of lines that are separated by .blas.ld elements (ld
	 * stands for leading dimension).
	 * NB: in case some filters were used, the leading dimension is not
	 * guaranteed to be the same in main memory (on the original matrix)
	 * and on the accelerator! */
	nxC = STARPU_MATRIX_GET_NX(descr[2]);
	nyC = STARPU_MATRIX_GET_NY(descr[2]);
	nyA = STARPU_MATRIX_GET_NY(descr[0]);

	ldA = STARPU_MATRIX_GET_LD(descr[0]);
	ldB = STARPU_MATRIX_GET_LD(descr[1]);
	ldC = STARPU_MATRIX_GET_LD(descr[2]);

	/* we assume a FORTRAN-ordering! */
	unsigned i,j,k;
	for (i = 0; i < nyC; i++)
	{
		for (j = 0; j < nxC; j++)
		{
			float sum = 0.0;

			for (k = 0; k < nyA; k++)
			{
				sum += subA[j+k*ldA]*subB[k+i*ldB];
			}

			subC[j + i*ldC] = sum;
		}
	}
}
