/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018       Alexis Juven
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
#include <stdio.h>
#include <string.h>
#include <starpu.h>

/*
 * The codelet is passed 3 matrices, the "descr" union-type field gives a
 * description of the layout of those 3 matrices in the local memory (ie. RAM
 * in the case of CPU, GPU frame buffer in the case of GPU etc.). Since we have
 * registered data with the "matrix" data interface, we use the matrix macros.
 */
void cpu_mult(void *descr[], void *cl_arg)
{
	int stride;
	float *subA, *subB, *subC;

	stride = *((int *)cl_arg);

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
	const uint32_t nxC = STARPU_MATRIX_GET_NX(descr[2]);
	const uint32_t nyC = STARPU_MATRIX_GET_NY(descr[2]);
	const uint32_t nyA = STARPU_MATRIX_GET_NY(descr[0]);

	const uint32_t ldA = STARPU_MATRIX_GET_LD(descr[0]);
	const uint32_t ldB = STARPU_MATRIX_GET_LD(descr[1]);
	const uint32_t ldC = STARPU_MATRIX_GET_LD(descr[2]);
	/* we assume a FORTRAN-ordering! */
	int i,j,k,ii,jj,kk;
	for (i = 0; i < nyC*nxC; i++) subC[i] = 0;
	//fprintf(stderr,"inside cpu_mult %dx%dx%d %d/%d on %d\n",nyC,nyA,nxC,starpu_worker_get_id(),STARPU_NMAXWORKERS,starpu_worker_get_devid(starpu_worker_get_id()));
	for (i=0;i<nyC;i+=stride)
	{
		for (k=0;k<nyA;k+=stride)
		{
			for (j=0;j<nxC;j+=stride)
			{
				for (ii = i; ii < i+stride; ii+=2)
				{
					float *sC0=subC+ii*ldC+j;
					float *sC1=subC+ii*ldC+ldC+j;
					for (kk = k; kk < k+stride; kk+=4)
					{
						float alpha00=subB[kk +  ii*ldB];
						float alpha01=subB[kk+1+ii*ldB];
						float alpha10=subB[kk+  ii*ldB+ldB];
						float alpha11=subB[kk+1+ii*ldB+ldB];
						float alpha02=subB[kk+2+ii*ldB];
						float alpha03=subB[kk+3+ii*ldB];
						float alpha12=subB[kk+2+ ii*ldB+ldB];
						float alpha13=subB[kk+3+ii*ldB+ldB];
						float *sA0=subA+kk*ldA+j;
						float *sA1=subA+kk*ldA+ldA+j;
						float *sA2=subA+kk*ldA+2*ldA+j;
						float *sA3=subA+kk*ldA+3*ldA+j;
						for (jj = 0; jj < stride; jj+=1)
						{
							sC0[jj] += alpha00*sA0[jj]+alpha01*sA1[jj]+alpha02*sA2[jj]+alpha03*sA3[jj];
							sC1[jj] += alpha10*sA0[jj]+alpha11*sA1[jj]+alpha12*sA2[jj]+alpha13*sA3[jj];
						}
					}
				}
			}
		}
	}
	//fprintf(stderr,"inside cpu_mult %dx%dx%d\n",nyC,nyA,nxC);
}

char* CPU = "cpu_mult";
char* GPU = "";
extern char *starpu_find_function(char *name, char *device)
{
	if (!strcmp(device,"gpu")) return GPU;
	return CPU;
}
