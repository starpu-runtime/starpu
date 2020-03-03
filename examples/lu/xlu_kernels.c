/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* LU Kernels */

#include "xlu.h"
#include <math.h>
#include <complex.h>

#ifdef STARPU_USE_CUDA
#include <cublas.h>
#include <starpu_cublas_v2.h>
#endif

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_LU_STR(name)  xstr(STARPU_LU(name))

#ifdef STARPU_USE_CUDA
static const TYPE p1 =  1.0f;
static const TYPE m1 = -1.0f;
#endif

/*
 *   U22
 */

static inline void STARPU_LU(common_u22)(void *descr[], int s, void *_args)
{
	(void)_args;
	TYPE *right 	= (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *left 	= (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *center 	= (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned dx = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned dy = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned dz = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ld12 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ld22 = STARPU_MATRIX_GET_LD(descr[2]);

#ifdef STARPU_USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif

	switch (s)
	{
		case 0:
			CPU_GEMM("N", "N", dy, dx, dz,
				(TYPE)-1.0, right, ld21, left, ld12,
				(TYPE)1.0, center, ld22);
			break;

#ifdef STARPU_USE_CUDA
		case 1:
		{
			status = CUBLAS_GEMM(starpu_cublas_get_local_handle(),
				CUBLAS_OP_N, CUBLAS_OP_N, dx, dy, dz,
				(CUBLAS_TYPE *)&m1, (CUBLAS_TYPE *)right, ld21, (CUBLAS_TYPE *)left, ld12,
				(CUBLAS_TYPE *)&p1, (CUBLAS_TYPE *)center, ld22);

			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_CUBLAS_REPORT_ERROR(status);

			break;
		}
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void STARPU_LU(cpu_u22)(void *descr[], void *_args)
{
	STARPU_LU(common_u22)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void STARPU_LU(cublas_u22)(void *descr[], void *_args)
{
	STARPU_LU(common_u22)(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

static struct starpu_perfmodel STARPU_LU(model_22) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_LU_STR(lu_model_22_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_LU_STR(lu_model_22_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_LU_STR(lu_model_22_openblas)
#else
	.symbol = STARPU_LU_STR(lu_model_22)
#endif
};

#ifdef STARPU_USE_CUDA
static int can_execute(unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	(void)task;
	(void)nimpl;
	enum starpu_worker_archtype type = starpu_worker_get_type(workerid);
	if (type == STARPU_CPU_WORKER || type == STARPU_MIC_WORKER)
		return 1;

#ifdef STARPU_SIMGRID
	/* We don't know, let's assume it can */
	return 1;
#else
	/* Cuda device */
	const struct cudaDeviceProp *props;
	props = starpu_cuda_get_device_properties(workerid);
	if (props->major >= 2 || props->minor >= 3)
	{
		/* At least compute capability 1.3, supports doubles */
		return 1;
	}
	else
	{
		/* Old card does not support doubles */
		return 0;
	}
#endif
}
#endif

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
struct starpu_codelet cl22 =
{
	.cpu_funcs = {STARPU_LU(cpu_u22)},
	.cpu_funcs_name = {STRINGIFY(STARPU_LU(cpu_u22))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_LU(cublas_u22)},
	CAN_EXECUTE
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &STARPU_LU(model_22)
};

/*
 * U12
 */

static inline void STARPU_LU(common_u12)(void *descr[], int s, void *_args)
{
	(void)_args;
	TYPE *sub11;
	TYPE *sub12;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	sub12 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld11 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld12 = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned nx12 = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny12 = STARPU_MATRIX_GET_NY(descr[1]);

#ifdef STARPU_USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif

	/* solve L11 U12 = A12 (find U12) */
	switch (s)
	{
		case 0:
			CPU_TRSM("L", "L", "N", "N", nx12, ny12,
					(TYPE)1.0, sub11, ld11, sub12, ld12);
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			status = CUBLAS_TRSM(starpu_cublas_get_local_handle(),
					CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
					ny12, nx12,
					(CUBLAS_TYPE*)&p1, (CUBLAS_TYPE*)sub11, ld11, (CUBLAS_TYPE*)sub12, ld12);

			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_CUBLAS_REPORT_ERROR(status);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void STARPU_LU(cpu_u12)(void *descr[], void *_args)
{
	STARPU_LU(common_u12)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void STARPU_LU(cublas_u12)(void *descr[], void *_args)
{
	STARPU_LU(common_u12)(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

static struct starpu_perfmodel STARPU_LU(model_12) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_LU_STR(lu_model_12_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_LU_STR(lu_model_12_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_LU_STR(lu_model_12_openblas)
#else
	.symbol = STARPU_LU_STR(lu_model_12)
#endif
};

struct starpu_codelet cl12 =
{
	.cpu_funcs = {STARPU_LU(cpu_u12)},
	.cpu_funcs_name = {STRINGIFY(STARPU_LU(cpu_u12))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_LU(cublas_u12)},
	CAN_EXECUTE
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &STARPU_LU(model_12)
};

/*
 * U21
 */

static inline void STARPU_LU(common_u21)(void *descr[], int s, void *_args)
{
	(void)_args;
	TYPE *sub11;
	TYPE *sub21;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	sub21 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld11 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned nx21 = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny21 = STARPU_MATRIX_GET_NY(descr[1]);

#ifdef STARPU_USE_CUDA
	cublasStatus status;
#endif

	switch (s)
	{
		case 0:
			CPU_TRSM("R", "U", "N", "U", nx21, ny21,
					(TYPE)1.0, sub11, ld11, sub21, ld21);
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			status = CUBLAS_TRSM(starpu_cublas_get_local_handle(),
					CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
					ny21, nx21,
					(CUBLAS_TYPE*)&p1, (CUBLAS_TYPE*)sub11, ld11, (CUBLAS_TYPE*)sub21, ld21);

			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_CUBLAS_REPORT_ERROR(status);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void STARPU_LU(cpu_u21)(void *descr[], void *_args)
{
	STARPU_LU(common_u21)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void STARPU_LU(cublas_u21)(void *descr[], void *_args)
{
	STARPU_LU(common_u21)(descr, 1, _args);
}
#endif

static struct starpu_perfmodel STARPU_LU(model_21) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_LU_STR(lu_model_21_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_LU_STR(lu_model_21_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_LU_STR(lu_model_21_openblas)
#else
	.symbol = STARPU_LU_STR(lu_model_21)
#endif
};

struct starpu_codelet cl21 =
{
	.cpu_funcs = {STARPU_LU(cpu_u21)},
	.cpu_funcs_name = {STRINGIFY(STARPU_LU(cpu_u21))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_LU(cublas_u21)},
	CAN_EXECUTE
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &STARPU_LU(model_21)
};

/*
 *	U11
 */

static inline void STARPU_LU(common_u11)(void *descr[], int s, void *_args)
{
	(void)_args;
	TYPE *sub11;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);

	unsigned long nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned long ld = STARPU_MATRIX_GET_LD(descr[0]);

	unsigned long z;

#ifdef STARPU_USE_CUDA
	cublasStatus status;
	cublasHandle_t handle;
	cudaStream_t stream;
#endif

	switch (s)
	{
		case 0:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				pivot = sub11[z+z*ld];
				STARPU_ASSERT(!ISZERO(pivot));

				CPU_SCAL(nx - z - 1, (1.0/pivot), &sub11[z+(z+1)*ld], ld);

				CPU_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			handle = starpu_cublas_get_local_handle();
			stream = starpu_cuda_get_local_stream();
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				TYPE inv_pivot;
				cudaMemcpyAsync(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				STARPU_ASSERT(!ISZERO(pivot));

				inv_pivot = 1.0/pivot;
				status = CUBLAS_SCAL(handle,
						nx - z - 1,
						(CUBLAS_TYPE*)&inv_pivot, (CUBLAS_TYPE*)&sub11[z+(z+1)*ld], ld);
				if (status != CUBLAS_STATUS_SUCCESS)
					STARPU_CUBLAS_REPORT_ERROR(status);

				status = CUBLAS_GER(handle,
						nx - z - 1, nx - z - 1,
						(CUBLAS_TYPE*)&m1,
						(CUBLAS_TYPE*)&sub11[(z+1)+z*ld], 1,
						(CUBLAS_TYPE*)&sub11[z+(z+1)*ld], ld,
						(CUBLAS_TYPE*)&sub11[(z+1) + (z+1)*ld],ld);
				if (status != CUBLAS_STATUS_SUCCESS)
					STARPU_CUBLAS_REPORT_ERROR(status);
			}

			cudaStreamSynchronize(stream);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void STARPU_LU(cpu_u11)(void *descr[], void *_args)
{
	STARPU_LU(common_u11)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void STARPU_LU(cublas_u11)(void *descr[], void *_args)
{
	STARPU_LU(common_u11)(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

static struct starpu_perfmodel STARPU_LU(model_11) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_LU_STR(lu_model_11_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_LU_STR(lu_model_11_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_LU_STR(lu_model_11_openblas)
#else
	.symbol = STARPU_LU_STR(lu_model_11)
#endif
};

struct starpu_codelet cl11 =
{
	.cpu_funcs = {STARPU_LU(cpu_u11)},
	.cpu_funcs_name = {STRINGIFY(STARPU_LU(cpu_u11))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_LU(cublas_u11)},
	CAN_EXECUTE
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &STARPU_LU(model_11)
};

/*
 *	U11 with pivoting
 */

static inline void STARPU_LU(common_u11_pivot)(void *descr[],
				int s, void *_args)
{
	TYPE *sub11;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);

	unsigned long nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned long ld = STARPU_MATRIX_GET_LD(descr[0]);

	unsigned long z;

	struct piv_s *piv = _args;
	unsigned *ipiv = piv->piv;
	unsigned first = piv->first;

#ifdef STARPU_USE_CUDA
	cublasStatus status;
	cublasHandle_t handle;
	cudaStream_t stream;
#endif

	switch (s)
	{
		case 0:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				pivot = sub11[z+z*ld];

				if (fabs((double)(pivot)) < PIVOT_THRESHHOLD)
				{

					/* find the pivot */
					int piv_ind = CPU_IAMAX(nx - z, &sub11[z*(ld+1)], ld);

					ipiv[z + first] = piv_ind + z + first;

					/* swap if needed */
					if (piv_ind != 0)
					{
						CPU_SWAP(nx, &sub11[z*ld], 1, &sub11[(z+piv_ind)*ld], 1);
					}

					pivot = sub11[z+z*ld];
				}

				STARPU_ASSERT(!ISZERO(pivot));

				CPU_SCAL(nx - z - 1, (1.0/pivot), &sub11[z+(z+1)*ld], ld);

				CPU_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}

			break;
#ifdef STARPU_USE_CUDA
		case 1:
			handle = starpu_cublas_get_local_handle();
			stream = starpu_cuda_get_local_stream();
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				TYPE inv_pivot;
				cudaMemcpyAsync(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);

				if (fabs((double)(pivot)) < PIVOT_THRESHHOLD)
				{
					/* find the pivot */
					int piv_ind;
					status = CUBLAS_IAMAX(handle,
						nx - z, (CUBLAS_TYPE*)&sub11[z*(ld+1)], ld, &piv_ind);
					piv_ind -= 1;
					if (status != CUBLAS_STATUS_SUCCESS)
						STARPU_CUBLAS_REPORT_ERROR(status);

					ipiv[z + first] = piv_ind + z + first;

					/* swap if needed */
					if (piv_ind != 0)
					{
						status = CUBLAS_SWAP(handle,
							nx,
							(CUBLAS_TYPE*)&sub11[z*ld], 1,
							(CUBLAS_TYPE*)&sub11[(z+piv_ind)*ld], 1);
						if (status != CUBLAS_STATUS_SUCCESS)
							STARPU_CUBLAS_REPORT_ERROR(status);
					}

					cudaMemcpyAsync(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost, stream);
					cudaStreamSynchronize(stream);
				}

				STARPU_ASSERT(!ISZERO(pivot));

				inv_pivot = 1.0/pivot;
				status = CUBLAS_SCAL(handle,
						nx - z - 1,
						(CUBLAS_TYPE*)&inv_pivot,
						(CUBLAS_TYPE*)&sub11[z+(z+1)*ld], ld);
				if (status != CUBLAS_STATUS_SUCCESS)
					STARPU_CUBLAS_REPORT_ERROR(status);

				status = CUBLAS_GER(handle,
						nx - z - 1, nx - z - 1,
						(CUBLAS_TYPE*)&m1,
						(CUBLAS_TYPE*)&sub11[(z+1)+z*ld], 1,
						(CUBLAS_TYPE*)&sub11[z+(z+1)*ld], ld,
						(CUBLAS_TYPE*)&sub11[(z+1) + (z+1)*ld],ld);
				if (status != CUBLAS_STATUS_SUCCESS)
						STARPU_CUBLAS_REPORT_ERROR(status);
			}

			cudaStreamSynchronize(stream);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void STARPU_LU(cpu_u11_pivot)(void *descr[], void *_args)
{
	STARPU_LU(common_u11_pivot)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void STARPU_LU(cublas_u11_pivot)(void *descr[], void *_args)
{
	STARPU_LU(common_u11_pivot)(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */

static struct starpu_perfmodel STARPU_LU(model_11_pivot) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_LU_STR(lu_model_11_pivot_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_LU_STR(lu_model_11_pivot_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_LU_STR(lu_model_11_pivot_openblas)
#else
	.symbol = STARPU_LU_STR(lu_model_11_pivot)
#endif
};

struct starpu_codelet cl11_pivot =
{
	.cpu_funcs = {STARPU_LU(cpu_u11_pivot)},
	.cpu_funcs_name = {STRINGIFY(STARPU_LU(cpu_u11_pivot))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_LU(cublas_u11_pivot)},
	CAN_EXECUTE
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &STARPU_LU(model_11_pivot)
};

/*
 *	Pivoting
 */

static inline void STARPU_LU(common_pivot)(void *descr[],
				int s, void *_args)
{
	TYPE *matrix;

	matrix = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	unsigned long nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned long ld = STARPU_MATRIX_GET_LD(descr[0]);

	unsigned row;

	struct piv_s *piv = _args;
	unsigned *ipiv = piv->piv;
	unsigned first = piv->first;

#ifdef STARPU_USE_CUDA
	cublasStatus status;
	cublasHandle_t handle;
#endif

	switch (s)
	{
		case 0:
			for (row = 0; row < nx; row++)
			{
				unsigned rowpiv = ipiv[row+first] - first;
				if (rowpiv != row)
				{
					CPU_SWAP(nx, &matrix[row*ld], 1, &matrix[rowpiv*ld], 1);
				}
			}
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			handle = starpu_cublas_get_local_handle();
			for (row = 0; row < nx; row++)
			{
				unsigned rowpiv = ipiv[row+first] - first;
				if (rowpiv != row)
				{
					status = CUBLAS_SWAP(handle,
							nx,
							(CUBLAS_TYPE*)&matrix[row*ld], 1,
							(CUBLAS_TYPE*)&matrix[rowpiv*ld], 1);
					if (status != CUBLAS_STATUS_SUCCESS)
						STARPU_CUBLAS_REPORT_ERROR(status);
				}
			}

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void STARPU_LU(cpu_pivot)(void *descr[], void *_args)
{
	STARPU_LU(common_pivot)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void STARPU_LU(cublas_pivot)(void *descr[], void *_args)
{
	STARPU_LU(common_pivot)(descr, 1, _args);
}

#endif /* STARPU_USE_CUDA */

static struct starpu_perfmodel STARPU_LU(model_pivot) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_LU_STR(lu_model_pivot_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_LU_STR(lu_model_pivot_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_LU_STR(lu_model_pivot_openblas)
#else
	.symbol = STARPU_LU_STR(lu_model_pivot)
#endif
};

struct starpu_codelet cl_pivot =
{
	.cpu_funcs = {STARPU_LU(cpu_pivot)},
	.cpu_funcs_name = {STRINGIFY(STARPU_LU(cpu_pivot))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_LU(cublas_pivot)},
	CAN_EXECUTE
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &STARPU_LU(model_pivot)
};
