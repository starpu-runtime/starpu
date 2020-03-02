/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * Standard kernels for the Cholesky factorization
 * U22 is the gemm update
 * U21 is the trsm update
 * U11 is the cholesky factorization
 */

#include <starpu.h>
#include "cholesky.h"
#include "../common/blas.h"
#if defined(STARPU_USE_CUDA)
#include <cublas.h>
#include <starpu_cublas_v2.h>
#if defined(STARPU_HAVE_MAGMA)
#include "magma.h"
#include "magma_lapack.h"
#endif
#endif

/*
 *   U22
 */

#if defined(STARPU_USE_CUDA)
static const float p1 =  1.0;
static const float m1 = -1.0;
#endif

static inline void chol_common_cpu_codelet_update_u22(void *descr[], int s, void *_args)
{
	(void)_args;
	/* printf("22\n"); */
	float *left 	= (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	float *right 	= (float *)STARPU_MATRIX_GET_PTR(descr[1]);
	float *center 	= (float *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned dx = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned dy = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned dz = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld12 = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ld22 = STARPU_MATRIX_GET_LD(descr[2]);

	if (s == 0)
	{
		int worker_size = starpu_combined_worker_get_size();

		if (worker_size == 1)
		{
			/* Sequential CPU kernel */
			STARPU_SGEMM("N", "T", dy, dx, dz, -1.0f, left, ld21,
				right, ld12, 1.0f, center, ld22);
		}
		else
		{
			/* Parallel CPU kernel */
			unsigned rank = starpu_combined_worker_get_rank();

			unsigned block_size = (dx + worker_size - 1)/worker_size;
			unsigned new_dx = STARPU_MIN(dx, block_size*(rank+1)) - block_size*rank;

			float *new_left = &left[block_size*rank];
			float *new_center = &center[block_size*rank];

			STARPU_SGEMM("N", "T", dy, new_dx, dz, -1.0f, new_left, ld21,
				right, ld12, 1.0f, new_center, ld22);
		}
	}
	else
	{
		/* CUDA kernel */
#ifdef STARPU_USE_CUDA
		cublasStatus_t status = cublasSgemm(starpu_cublas_get_local_handle(),
				CUBLAS_OP_N, CUBLAS_OP_T, dy, dx, dz,
				&m1, left, ld21, right, ld12,
				&p1, center, ld22);
		if (status != CUBLAS_STATUS_SUCCESS)
			STARPU_CUBLAS_REPORT_ERROR(status);
#endif

	}
}

void chol_cpu_codelet_update_u22(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_u22(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_u22(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_u22(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

/*
 * U21
 */

static inline void chol_common_codelet_update_u21(void *descr[], int s, void *_args)
{
/*	printf("21\n"); */
	float *sub11;
	float *sub21;
	(void)_args;

	sub11 = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	sub21 = (float *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld11 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned nx21 = STARPU_MATRIX_GET_NY(descr[1]);
	unsigned ny21 = STARPU_MATRIX_GET_NX(descr[1]);

#ifdef STARPU_USE_CUDA
	cublasStatus status;
#endif

	switch (s)
	{
		case 0:
			STARPU_STRSM("R", "L", "T", "N", nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			status = cublasStrsm(starpu_cublas_get_local_handle(),
					CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
					nx21, ny21, &p1, sub11, ld11, sub21, ld21);
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_CUBLAS_REPORT_ERROR(status);
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void chol_cpu_codelet_update_u21(void *descr[], void *_args)
{
	 chol_common_codelet_update_u21(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_u21(void *descr[], void *_args)
{
	chol_common_codelet_update_u21(descr, 1, _args);
}
#endif

/*
 *	U11
 */

static inline void chol_common_codelet_update_u11(void *descr[], int s, void *_args)
{
/*	printf("11\n"); */
	float *sub11;
	(void)_args;

	sub11 = (float *)STARPU_MATRIX_GET_PTR(descr[0]);

	unsigned nx = STARPU_MATRIX_GET_NY(descr[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(descr[0]);

	unsigned z;

	switch (s)
	{
		case 0:

#ifdef STARPU_MKL
			STARPU_SPOTRF("L", nx, sub11, ld);
#else
			/*
			 *	- alpha 11 <- lambda 11 = sqrt(alpha11)
			 *	- alpha 21 <- l 21	= alpha 21 / lambda 11
			 *	- A22 <- A22 - l21 trans(l21)
			 */

			for (z = 0; z < nx; z++)
			{
				float lambda11;
				lambda11 = sqrt(sub11[z+z*ld]);
				sub11[z+z*ld] = lambda11;

				STARPU_ASSERT(lambda11 != 0.0f);

				STARPU_SSCAL(nx - z - 1, 1.0f/lambda11, &sub11[(z+1)+z*ld], 1);

				STARPU_SSYR("L", nx - z - 1, -1.0f,
							&sub11[(z+1)+z*ld], 1,
							&sub11[(z+1)+(z+1)*ld], ld);
			}
#endif
			break;
#ifdef STARPU_USE_CUDA
		case 1:
#ifdef STARPU_HAVE_MAGMA
			{
			int ret;
			int info;
			cudaStream_t stream = starpu_cuda_get_local_stream();
#if (MAGMA_VERSION_MAJOR > 1) || (MAGMA_VERSION_MAJOR == 1 && MAGMA_VERSION_MINOR >= 4)
			cublasSetKernelStream(stream);
			magmablasSetKernelStream(stream);
#else
			starpu_cublas_set_stream();
#endif
			ret = magma_spotrf_gpu(MagmaLower, nx, sub11, ld, &info);
			if (ret != MAGMA_SUCCESS)
			{
				fprintf(stderr, "Error in Magma: %d\n", ret);
				STARPU_ABORT();
			}
#if (MAGMA_VERSION_MAJOR > 1) || (MAGMA_VERSION_MAJOR == 1 && MAGMA_VERSION_MINOR >= 4)
			cudaError_t cures = cudaStreamSynchronize(stream);
#else
			cudaError_t cures = cudaDeviceSynchronize();
#endif
			STARPU_ASSERT(!cures);
			}
#else
			{

			float *lambda11;
			cublasStatus_t status;
			cudaStream_t stream = starpu_cuda_get_local_stream();
			cublasHandle_t handle = starpu_cublas_get_local_handle();
			cudaHostAlloc((void **)&lambda11, sizeof(float), 0);

			for (z = 0; z < nx; z++)
			{
				cudaMemcpyAsync(lambda11, &sub11[z+z*ld], sizeof(float), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);

				STARPU_ASSERT(*lambda11 != 0.0f);

				*lambda11 = sqrt(*lambda11);

/*				cublasSetVector(1, sizeof(float), lambda11, sizeof(float), &sub11[z+z*ld], sizeof(float)); */
				cudaMemcpyAsync(&sub11[z+z*ld], lambda11, sizeof(float), cudaMemcpyHostToDevice, stream);
				float scal = 1.0f/(*lambda11);

				status = cublasSscal(handle,
						     nx - z - 1, &scal, &sub11[(z+1)+z*ld], 1);
				if (status != CUBLAS_STATUS_SUCCESS)
					STARPU_CUBLAS_REPORT_ERROR(status);

				status = cublasSsyr(handle,
						    CUBLAS_FILL_MODE_UPPER,
						    nx - z - 1, &m1,
						    &sub11[(z+1)+z*ld], 1,
						    &sub11[(z+1)+(z+1)*ld], ld);
				if (status != CUBLAS_STATUS_SUCCESS)
					STARPU_CUBLAS_REPORT_ERROR(status);
			}

			cudaStreamSynchronize(stream);
			cudaFreeHost(lambda11);
			}
#endif
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}


void chol_cpu_codelet_update_u11(void *descr[], void *_args)
{
	chol_common_codelet_update_u11(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_u11(void *descr[], void *_args)
{
	chol_common_codelet_update_u11(descr, 1, _args);
}
#endif/* STARPU_USE_CUDA */

struct starpu_perfmodel chol_model_11;
struct starpu_perfmodel chol_model_21;
struct starpu_perfmodel chol_model_22;

struct starpu_codelet cl11 =
{
	.type = STARPU_SEQ,
	.cpu_funcs = {chol_cpu_codelet_update_u11},
	.cpu_funcs_name = {"chol_cpu_codelet_update_u11"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u11},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.model = &chol_model_11,
	.color = 0xffff00,
};

struct starpu_codelet cl21 =
{
	.type = STARPU_SEQ,
	.cpu_funcs = {chol_cpu_codelet_update_u21},
	.cpu_funcs_name = {"chol_cpu_codelet_update_u21"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u21},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_RW },
	.model = &chol_model_21,
	.color = 0x8080ff,
};

struct starpu_codelet cl22 =
{
	.type = STARPU_SEQ,
	.max_parallelism = INT_MAX,
	.cpu_funcs = {chol_cpu_codelet_update_u22},
	.cpu_funcs_name = {"chol_cpu_codelet_update_u22"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u22},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.model = &chol_model_22,
	.color = 0x00ff00,
};

struct starpu_codelet cl11_gpu =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u11},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.model = &chol_model_11,
	.color = 0xffff00,
};

struct starpu_codelet cl21_gpu =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u21},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_RW },
	.model = &chol_model_21,
	.color = 0x8080ff,
};

struct starpu_codelet cl22_gpu =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u22},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.model = &chol_model_22,
	.color = 0x00ff00,
};

struct starpu_codelet cl11_cpu =
{
	.type = STARPU_SEQ,
	.cpu_funcs = {chol_cpu_codelet_update_u11},
	.cpu_funcs_name = {"chol_cpu_codelet_update_u11"},
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.model = &chol_model_11,
	.color = 0xffff00,
};

struct starpu_codelet cl21_cpu =
{
	.type = STARPU_SEQ,
	.cpu_funcs = {chol_cpu_codelet_update_u21},
	.cpu_funcs_name = {"chol_cpu_codelet_update_u21"},
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_RW },
	.model = &chol_model_21,
	.color = 0x8080ff,
};

struct starpu_codelet cl22_cpu =
{
	.type = STARPU_SEQ,
	.max_parallelism = INT_MAX,
	.cpu_funcs = {chol_cpu_codelet_update_u22},
	.cpu_funcs_name = {"chol_cpu_codelet_update_u22"},
	.nbuffers = 3,
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.model = &chol_model_22,
	.color = 0x00ff00,
};

