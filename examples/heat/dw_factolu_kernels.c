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

unsigned count_11_per_worker[STARPU_NMAXWORKERS] = {0};
unsigned count_12_per_worker[STARPU_NMAXWORKERS] = {0};
unsigned count_21_per_worker[STARPU_NMAXWORKERS] = {0};
unsigned count_22_per_worker[STARPU_NMAXWORKERS] = {0};

unsigned count_total_per_worker[STARPU_NMAXWORKERS] = {0};

unsigned count_11_total = 0;
unsigned count_12_total = 0;
unsigned count_21_total = 0;
unsigned count_22_total = 0;

void display_stat_heat(void)
{
	unsigned nworkers = starpu_get_worker_count();

	fprintf(stderr, "STATS : \n");

	unsigned worker;
	for (worker = 0; worker < nworkers; worker++)
	{
		count_total_per_worker[worker] = count_11_per_worker[worker] 
					+ count_12_per_worker[worker]
					+ count_21_per_worker[worker]
					+ count_22_per_worker[worker];

		count_11_total += count_11_per_worker[worker];
		count_12_total += count_12_per_worker[worker];
		count_21_total += count_21_per_worker[worker];
		count_22_total += count_22_per_worker[worker];
	}

	fprintf(stderr, "\t11 (diagonal block LU)\n");
	for (worker = 0; worker < nworkers; worker++)
	{
		if (count_total_per_worker[worker])
		{
			char name[32];
			starpu_get_worker_name(worker, name, 32);
			
			fprintf(stderr, "\t\t%s -> %d / %d (%2.2f %%)\n", name, count_11_per_worker[worker], count_11_total, (100.0*count_11_per_worker[worker])/count_11_total);
		}
	}

	fprintf(stderr, "\t12 (TRSM)\n");
	for (worker = 0; worker < nworkers; worker++)
	{
		if (count_total_per_worker[worker])
		{
			char name[32];
			starpu_get_worker_name(worker, name, 32);
			
			fprintf(stderr, "\t\t%s -> %d / %d (%2.2f %%)\n", name, count_12_per_worker[worker], count_12_total, (100.0*count_12_per_worker[worker])/count_12_total);
		}
	}
	
	
	fprintf(stderr, "\t21 (TRSM)\n");
	for (worker = 0; worker < nworkers; worker++)
	{
		if (count_total_per_worker[worker])
		{
			char name[32];
			starpu_get_worker_name(worker, name, 32);
			
			fprintf(stderr, "\t\t%s -> %d / %d (%2.2f %%)\n", name, count_21_per_worker[worker], count_21_total, (100.0*count_21_per_worker[worker])/count_21_total);
		}
	}
	
	fprintf(stderr, "\t22 (SGEMM)\n");
	for (worker = 0; worker < nworkers; worker++)
	{
		if (count_total_per_worker[worker])
		{
			char name[32];
			starpu_get_worker_name(worker, name, 32);
			
			fprintf(stderr, "\t\t%s -> %d / %d (%2.2f %%)\n", name, count_22_per_worker[worker], count_22_total, (100.0*count_22_per_worker[worker])/count_22_total);
		}
	}
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

			cudaThreadSynchronize();

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

	int id = starpu_get_worker_id();
	count_22_per_worker[id]++;
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u22(starpu_data_interface_t *descr, void *_args)
{
	dw_common_core_codelet_update_u22(descr, 1, _args);

	int id = starpu_get_worker_id();
	count_22_per_worker[id]++;
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

			cudaThreadSynchronize();

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

	int id = starpu_get_worker_id();
	count_12_per_worker[id]++;
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u12(starpu_data_interface_t *descr, void *_args)
{
	 dw_common_codelet_update_u12(descr, 1, _args);

	int id = starpu_get_worker_id();
	count_12_per_worker[id]++;
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

			cudaThreadSynchronize();

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

	int id = starpu_get_worker_id();
	count_21_per_worker[id]++;
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u21(starpu_data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u21(descr, 1, _args);

	int id = starpu_get_worker_id();
	count_21_per_worker[id]++;
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
			fprintf(stderr, "%2.2f\t", tab[(size_t)j+(size_t)i*ld]);
		}
		fprintf(stderr, "\n");
	}
	
	fprintf(stderr, "\n");
}

static inline void dw_common_codelet_update_u11(starpu_data_interface_t *descr, int s, __attribute__((unused)) void *_args) 
{
	float *sub11;

	sub11 = (float *)descr[0].blas.ptr; 

	unsigned long nx = descr[0].blas.nx;
	unsigned long ld = descr[0].blas.ld;

	unsigned long z;

	float pouet;

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
				cudaMemcpy(&pivot, &sub11[z+z*ld], sizeof(float), cudaMemcpyDeviceToHost);
				cudaStreamSynchronize(0);

				STARPU_ASSERT(pivot != 0.0f);
				
				cublasSscal(nx - z - 1, 1.0f/pivot, &sub11[z+(z+1)*ld], ld);
				
				cublasSger(nx - z - 1, nx - z - 1, -1.0f,
								&sub11[z+(z+1)*ld], ld,
								&sub11[(z+1)+z*ld], 1,
								&sub11[(z+1) + (z+1)*ld],ld);
			}

			cudaThreadSynchronize();

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

	int id = starpu_get_worker_id();
	count_11_per_worker[id]++;
}

#ifdef USE_CUDA
void dw_cublas_codelet_update_u11(starpu_data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u11(descr, 1, _args);

	int id = starpu_get_worker_id();
	count_11_per_worker[id]++;
}
#endif// USE_CUDA
