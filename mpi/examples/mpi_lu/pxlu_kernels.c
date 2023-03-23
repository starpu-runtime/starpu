/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "pxlu.h"
#include "pxlu_kernels.h"
#include <math.h>

///#define VERBOSE_KERNELS	1

#ifdef STARPU_USE_CUDA
static const TYPE p1 =  1.0f;
static const TYPE m1 = -1.0f;
#endif

/*
 * GEMM
 */

static inline void STARPU_PLU(common_gemm)(void *descr[], int s, void *_args)
{
	TYPE *right 	= (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *left 	= (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *center 	= (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned dx = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned dy = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned dz = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ld12 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ld22 = STARPU_MATRIX_GET_LD(descr[2]);

#ifdef VERBOSE_KERNELS
	struct debug_info *info = _args;

	int rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	fprintf(stderr, "KERNEL GEMM %d - k = %u i = %u j = %u\n", rank, info->k, info->i, info->j);
#else
	(void)_args;
#endif

#ifdef STARPU_USE_CUDA
	cublasStatus_t status;
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
#ifdef VERBOSE_KERNELS
	fprintf(stderr, "KERNEL GEMM %d - k = %u i = %u j = %u done\n", rank, info->k, info->i, info->j);
#endif
}

static void STARPU_PLU(cpu_gemm)(void *descr[], void *_args)
{
	STARPU_PLU(common_gemm)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_gemm)(void *descr[], void *_args)
{
	STARPU_PLU(common_gemm)(descr, 1, _args);
}
#endif// STARPU_USE_CUDA

static struct starpu_perfmodel STARPU_PLU(model_gemm) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_gemm_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_gemm_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_gemm_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_gemm)
#endif
};

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
struct starpu_codelet STARPU_PLU(cl_gemm) =
{
	.cpu_funcs = {STARPU_PLU(cpu_gemm)},
	.cpu_funcs_name = {STRINGIFY(STARPU_PLU(cpu_gemm))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_gemm)},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &STARPU_PLU(model_gemm)
};

/*
 * TRSM_LL
 */

static inline void STARPU_PLU(common_trsmll)(void *descr[], int s, void *_args)
{
	TYPE *sub11;
	TYPE *sub12;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	sub12 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld11 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld12 = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned nx12 = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny12 = STARPU_MATRIX_GET_NY(descr[1]);

#ifdef VERBOSE_KERNELS
	struct debug_info *info = _args;

	int rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
#warning fixed debugging according to other tweak
	//fprintf(stderr, "KERNEL TRSM_LL %d - k = %u i %u\n", rank, info->k, info->i);
	fprintf(stderr, "KERNEL TRSM_RU %d - k = %u i %u\n", rank, info->k, info->j);

	//fprintf(stderr, "INPUT 12 GETRF\n");
	fprintf(stderr, "INPUT 21 GETRF\n");
	STARPU_PLU(display_data_content)(sub11, nx12);
	//fprintf(stderr, "INPUT 12 TRSM_LL\n");
	fprintf(stderr, "INPUT 21 TRSM_RU\n");
	STARPU_PLU(display_data_content)(sub12, nx12);
#else
	(void)_args;
#endif

#ifdef STARPU_USE_CUDA
	cublasStatus_t status;
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

#ifdef VERBOSE_KERNELS
	//fprintf(stderr, "OUTPUT 12 TRSM_LL\n");
	fprintf(stderr, "OUTPUT 21 TRSM_RU\n");
	STARPU_PLU(display_data_content)(sub12, nx12);
#endif
}

static void STARPU_PLU(cpu_trsmll)(void *descr[], void *_args)
{
	STARPU_PLU(common_trsmll)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_trsmll)(void *descr[], void *_args)
{
	STARPU_PLU(common_trsmll)(descr, 1, _args);
}
#endif // STARPU_USE_CUDA

static struct starpu_perfmodel STARPU_PLU(model_trsm_ll) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_trsm_ll_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_trsm_ll_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_trsm_ll_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_trsm_ll)
#endif
};

struct starpu_codelet STARPU_PLU(cl_trsm_ll) =
{
	.cpu_funcs = {STARPU_PLU(cpu_trsmll)},
	.cpu_funcs_name = {STRINGIFY(STARPU_PLU(cpu_trsmll))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_trsmll)},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &STARPU_PLU(model_trsm_ll)
};

/*
 * TRSM_RU
 */

static inline void STARPU_PLU(common_trsmru)(void *descr[], int s, void *_args)
{
	TYPE *sub11;
	TYPE *sub21;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	sub21 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);

	unsigned ld11 = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ld21 = STARPU_MATRIX_GET_LD(descr[1]);

	unsigned nx21 = STARPU_MATRIX_GET_NX(descr[1]);
	unsigned ny21 = STARPU_MATRIX_GET_NY(descr[1]);

#ifdef VERBOSE_KERNELS
	struct debug_info *info = _args;

	int rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
#warning fixed debugging according to other tweak
	//fprintf(stderr, "KERNEL TRSM_RU %d (k = %u, i = %u)\n", rank, info->k, info->i);
	fprintf(stderr, "KERNEL TRSM_LL %d (k = %u, j = %u)\n", rank, info->k, info->j);

	//fprintf(stderr, "INPUT 21 GETRF\n");
	fprintf(stderr, "INPUT 12 GETRF\n");
	STARPU_PLU(display_data_content)(sub11, nx21);
	//fprintf(stderr, "INPUT 21 TRSM_RU\n");
	fprintf(stderr, "INPUT 12 TRSM_LL\n");
	STARPU_PLU(display_data_content)(sub21, nx21);
#else
	(void)_args;
#endif

#ifdef STARPU_USE_CUDA
	cublasStatus_t status;
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

#ifdef VERBOSE_KERNELS
	//fprintf(stderr, "OUTPUT 21 GETRF\n");
	fprintf(stderr, "OUTPUT 12 GETRF\n");
	STARPU_PLU(display_data_content)(sub11, nx21);
	//fprintf(stderr, "OUTPUT 21 TRSM_RU\n");
	fprintf(stderr, "OUTPUT 12 TRSM_LL\n");
	STARPU_PLU(display_data_content)(sub21, nx21);
#endif
}

static void STARPU_PLU(cpu_trsmru)(void *descr[], void *_args)
{
	STARPU_PLU(common_trsmru)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_trsmru)(void *descr[], void *_args)
{
	STARPU_PLU(common_trsmru)(descr, 1, _args);
}
#endif

static struct starpu_perfmodel STARPU_PLU(model_trsm_ru) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_trsm_ru_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_trsm_ru_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_trsm_ru_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_trsm_ru)
#endif
};

struct starpu_codelet STARPU_PLU(cl_trsm_ru) =
{
	.cpu_funcs = {STARPU_PLU(cpu_trsmru)},
	.cpu_funcs_name = {STRINGIFY(STARPU_PLU(cpu_trsmru))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_trsmru)},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &STARPU_PLU(model_trsm_ru)
};


/*
 *	GETRF
 */

static inline void STARPU_PLU(common_getrf)(void *descr[], int s, void *_args)
{
	TYPE *sub11;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);

	unsigned long nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned long ld = STARPU_MATRIX_GET_LD(descr[0]);

	unsigned long z;

#ifdef STARPU_USE_CUDA
	cublasStatus_t status;
	cublasHandle_t handle;
	cudaStream_t stream;
#endif

#ifdef VERBOSE_KERNELS
	struct debug_info *info = _args;

	int rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	fprintf(stderr, "KERNEL 11 %d - k = %u\n", rank, info->k);
#else
	(void)_args;
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
#ifdef VERBOSE_KERNELS
	fprintf(stderr, "KERNEL GETRF %d - k = %u\n", rank, info->k);
#endif
}

static void STARPU_PLU(cpu_getrf)(void *descr[], void *_args)
{
	STARPU_PLU(common_getrf)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_getrf)(void *descr[], void *_args)
{
	STARPU_PLU(common_getrf)(descr, 1, _args);
}
#endif// STARPU_USE_CUDA

static struct starpu_perfmodel STARPU_PLU(model_getrf) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_getrf_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_getrf_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_getrf_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_getrf)
#endif
};

struct starpu_codelet STARPU_PLU(cl_getrf) =
{
	.cpu_funcs = {STARPU_PLU(cpu_getrf)},
	.cpu_funcs_name = {STRINGIFY(STARPU_PLU(cpu_getrf))},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_getrf)},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &STARPU_PLU(model_getrf)
};
