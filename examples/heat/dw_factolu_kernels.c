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

#include "dw_factolu.h"

unsigned count_11_core = 0;
unsigned count_12_core = 0;
unsigned count_21_core = 0;
unsigned count_22_core = 0;

unsigned count_11_cublas = 0;
unsigned count_12_cublas = 0;
unsigned count_21_cublas = 0;
unsigned count_22_cublas = 0;

void display_stat_heat(void)
{
	fprintf(stderr, "STATS : \n");
	fprintf(stderr, "11 : core %d (%2.2f) cublas %d (%2.2f)\n", count_11_core, (100.0*count_11_core)/(count_11_core+count_11_cublas), count_11_cublas, (100.0*count_11_cublas)/(count_11_core+count_11_cublas));
	fprintf(stderr, "12 : core %d (%2.2f) cublas %d (%2.2f)\n", count_12_core, (100.0*count_12_core)/(count_12_core+count_12_cublas), count_12_cublas, (100.0*count_12_cublas)/(count_12_core+count_12_cublas));
	fprintf(stderr, "21 : core %d (%2.2f) cublas %d (%2.2f)\n", count_21_core, (100.0*count_21_core)/(count_21_core+count_21_cublas), count_21_cublas, (100.0*count_21_cublas)/(count_21_core+count_21_cublas));
	fprintf(stderr, "22 : core %d (%2.2f) cublas %d (%2.2f)\n", count_22_core, (100.0*count_22_core)/(count_22_core+count_22_cublas), count_22_cublas, (100.0*count_22_cublas)/(count_22_core+count_22_cublas));
}

/*
 *   U22 
 */

static inline void dw_common_core_codelet_update_u22(starpu_data_interface_t *buffers, int s, __attribute__((unused)) void *_args)
{
	float *left 	= (float *)buffers[0].blas.ptr;
	float *right 	= (float *)buffers[1].blas.ptr;
	float *center 	= (float *)buffers[2].blas.ptr;

	unsigned dx = buffers[2].blas.nx;
	unsigned dy = buffers[2].blas.ny;
	unsigned dz = buffers[0].blas.ny;

	unsigned ld12 = buffers[0].blas.ld;
	unsigned ld21 = buffers[1].blas.ld;
	unsigned ld22 = buffers[2].blas.ld;

#ifdef USE_CUDA
	cublasStatus status;
#endif

	switch (s) {
		case 0:
			SGEMM("N", "N",	dy, dx, dz, 
				-1.0f, left, ld21, right, ld12,
					     1.0f, center, ld22);
			break;

#ifdef USE_CUDA
		case 1:
			cublasSgemm('n', 'n', dx, dy, dz, -1.0f, left, ld21,
					right, ld12, 1.0f, center, ld22);
			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_ASSERT(0);

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void dw_core_codelet_update_u22(starpu_data_interface_t *descr, void *_args)
{
	dw_common_core_codelet_update_u22(descr, 0, _args);
	(void)STARPU_ATOMIC_ADD(&count_22_core, 1);
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u22(starpu_data_interface_t *descr, void *_args)
{
	dw_common_core_codelet_update_u22(descr, 1, _args);
	(void)STARPU_ATOMIC_ADD(&count_22_cublas, 1);
}
#endif// USE_CUDA

/*
 * U12
 */

static inline void dw_common_codelet_update_u12(starpu_data_interface_t *buffers, int s, __attribute__((unused)) void *_args) {
	float *sub11;
	float *sub12;

	sub11 = (float *)buffers[0].blas.ptr;	
	sub12 = (float *)buffers[1].blas.ptr;

	unsigned ld11 = buffers[0].blas.ld;
	unsigned ld12 = buffers[1].blas.ld;

	unsigned nx12 = buffers[1].blas.nx;
	unsigned ny12 = buffers[1].blas.ny;
	
#ifdef USE_CUDA
	cublasStatus status;
#endif

	/* solve L11 U12 = A12 (find U12) */
	switch (s) {
		case 0:
			STRSM("L", "L", "N", "N",
					 nx12, ny12, 1.0f, sub11, ld11, sub12, ld12);
			break;
#ifdef USE_CUDA
		case 1:
			cublasStrsm('L', 'L', 'N', 'N', ny12, nx12,
					1.0f, sub11, ld11, sub12, ld12);
			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_ASSERT(0);

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void dw_core_codelet_update_u12(starpu_data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u12(descr, 0, _args);
	(void)STARPU_ATOMIC_ADD(&count_12_core, 1);
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u12(starpu_data_interface_t *descr, void *_args)
{
	 dw_common_codelet_update_u12(descr, 1, _args);
	(void)STARPU_ATOMIC_ADD(&count_12_cublas, 1);
}
#endif // USE_CUDA

/* 
 * U21
 */

static inline void dw_common_codelet_update_u21(starpu_data_interface_t *buffers, int s, __attribute__((unused)) void *_args) {
	float *sub11;
	float *sub21;

	sub11 = (float *)buffers[0].blas.ptr;
	sub21 = (float *)buffers[1].blas.ptr;

	unsigned ld11 = buffers[0].blas.ld;
	unsigned ld21 = buffers[1].blas.ld;

	unsigned nx21 = buffers[1].blas.nx;
	unsigned ny21 = buffers[1].blas.ny;
	
#ifdef USE_CUDA
	cublasStatus status;
#endif

	switch (s) {
		case 0:
			STRSM("R", "U", "N", "U", nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#ifdef USE_CUDA
		case 1:
			cublasStrsm('R', 'U', 'N', 'U', ny21, nx21, 1.0f, sub11, ld11, sub21, ld21);
			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_ASSERT(0);

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void dw_core_codelet_update_u21(starpu_data_interface_t *descr, void *_args)
{
	 dw_common_codelet_update_u21(descr, 0, _args);
	(void)STARPU_ATOMIC_ADD(&count_21_core, 1);
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u21(starpu_data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u21(descr, 1, _args);
	(void)STARPU_ATOMIC_ADD(&count_21_cublas, 1);
}
#endif 

/*
 *	U11
 */

static inline void debug_print(float *tab, unsigned ld, unsigned n)
{
	unsigned j,i;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < n; i++)
		{
			fprintf(stderr, "%2.2f\t", tab[j+i*ld]);
		}
		fprintf(stderr, "\n");
	}
	
	fprintf(stderr, "\n");
}

static inline void dw_common_codelet_update_u11(starpu_data_interface_t *descr, int s, __attribute__((unused)) void *_args) 
{
	float *sub11;

	sub11 = (float *)descr[0].blas.ptr; 

	unsigned nx = descr[0].blas.nx;
	unsigned ld = descr[0].blas.ld;

	unsigned z;

	switch (s) {
		case 0:
			for (z = 0; z < nx; z++)
			{
				float pivot;
				pivot = sub11[z+z*ld];
				STARPU_ASSERT(pivot != 0.0f);
		
				SSCAL(nx - z - 1, (1.0f/pivot), &sub11[z+(z+1)*ld], ld);
		
				SGER(nx - z - 1, nx - z - 1, -1.0f,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1)+z*ld], 1,
						&sub11[(z+1) + (z+1)*ld],ld);
			}
			break;
#ifdef USE_CUDA
		case 1:
			for (z = 0; z < nx; z++)
			{
				float pivot;
				/* ok that's dirty and ridiculous ... */
				cublasGetVector(1, sizeof(float), &sub11[z+z*ld], sizeof(float), &pivot, sizeof(float));

				STARPU_ASSERT(pivot != 0.0f);
				
				cublasSscal(nx - z - 1, 1.0f/pivot, &sub11[z+(z+1)*ld], ld);
				
				cublasSger(nx - z - 1, nx - z - 1, -1.0f,
								&sub11[z+(z+1)*ld], ld,
								&sub11[(z+1)+z*ld], 1,
								&sub11[(z+1) + (z+1)*ld],ld);
			}
			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}

}


void dw_core_codelet_update_u11(starpu_data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u11(descr, 0, _args);
	(void)STARPU_ATOMIC_ADD(&count_11_core, 1);
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u11(starpu_data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u11(descr, 1, _args);
	(void)STARPU_ATOMIC_ADD(&count_11_cublas, 1);
}
#endif// USE_CUDA
