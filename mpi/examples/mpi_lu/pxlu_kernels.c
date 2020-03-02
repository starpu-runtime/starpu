/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * U22
 */

static inline void STARPU_PLU(common_u22)(void *descr[], int s, void *_args)
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
	fprintf(stderr, "KERNEL 22 %d - k = %u i = %u j = %u\n", rank, info->k, info->i, info->j);
#else
	(void)_args;
#endif

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
			CUBLAS_GEMM('n', 'n', dx, dy, dz,
				(TYPE)-1.0, right, ld21, left, ld12,
				(TYPE)1.0f, center, ld22);

			status = cublasGetError();
			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_CUBLAS_REPORT_ERROR(status);

			if (STARPU_UNLIKELY((cures = cudaStreamSynchronize(starpu_cuda_get_local_stream())) != cudaSuccess))
				STARPU_CUDA_REPORT_ERROR(cures);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
#ifdef VERBOSE_KERNELS
	fprintf(stderr, "KERNEL 22 %d - k = %u i = %u j = %u done\n", rank, info->k, info->i, info->j);
#endif
}

static void STARPU_PLU(cpu_u22)(void *descr[], void *_args)
{
	STARPU_PLU(common_u22)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_u22)(void *descr[], void *_args)
{
	STARPU_PLU(common_u22)(descr, 1, _args);
}
#endif// STARPU_USE_CUDA

static struct starpu_perfmodel STARPU_PLU(model_22) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_22_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_22_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_22_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_22)
#endif
};

struct starpu_codelet STARPU_PLU(cl22) =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {STARPU_PLU(cpu_u22)},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_u22)},
#endif
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &STARPU_PLU(model_22)
};


/*
 * U12
 */

static inline void STARPU_PLU(common_u12)(void *descr[], int s, void *_args)
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
	//fprintf(stderr, "KERNEL 12 %d - k = %u i %u\n", rank, info->k, info->i);
	fprintf(stderr, "KERNEL 21 %d - k = %u i %u\n", rank, info->k, info->j);

	//fprintf(stderr, "INPUT 12 U11\n");
	fprintf(stderr, "INPUT 21 U11\n");
	STARPU_PLU(display_data_content)(sub11, nx12);
	//fprintf(stderr, "INPUT 12 U12\n");
	fprintf(stderr, "INPUT 21 U21\n");
	STARPU_PLU(display_data_content)(sub12, nx12);
#else
	(void)_args;
#endif

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
			CUBLAS_TRSM('L', 'L', 'N', 'N', ny12, nx12,
					(TYPE)1.0, sub11, ld11, sub12, ld12);

			status = cublasGetError();
			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_CUBLAS_REPORT_ERROR(status);

			if (STARPU_UNLIKELY((cures = cudaStreamSynchronize(starpu_cuda_get_local_stream())) != cudaSuccess))
				STARPU_CUDA_REPORT_ERROR(cures);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}

#ifdef VERBOSE_KERNELS
	//fprintf(stderr, "OUTPUT 12 U12\n");
	fprintf(stderr, "OUTPUT 21 U21\n");
	STARPU_PLU(display_data_content)(sub12, nx12);
#endif
}

static void STARPU_PLU(cpu_u12)(void *descr[], void *_args)
{
	STARPU_PLU(common_u12)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_u12)(void *descr[], void *_args)
{
	STARPU_PLU(common_u12)(descr, 1, _args);
}
#endif // STARPU_USE_CUDA

static struct starpu_perfmodel STARPU_PLU(model_12) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_12_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_12_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_12_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_12)
#endif
};

struct starpu_codelet STARPU_PLU(cl12) =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {STARPU_PLU(cpu_u12)},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_u12)},
#endif
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &STARPU_PLU(model_12)
};


/*
 * U21
 */

static inline void STARPU_PLU(common_u21)(void *descr[], int s, void *_args)
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
	//fprintf(stderr, "KERNEL 21 %d (k = %u, i = %u)\n", rank, info->k, info->i);
	fprintf(stderr, "KERNEL 12 %d (k = %u, j = %u)\n", rank, info->k, info->j);

	//fprintf(stderr, "INPUT 21 U11\n");
	fprintf(stderr, "INPUT 12 U11\n");
	STARPU_PLU(display_data_content)(sub11, nx21);
	//fprintf(stderr, "INPUT 21 U21\n");
	fprintf(stderr, "INPUT 12 U12\n");
	STARPU_PLU(display_data_content)(sub21, nx21);
#else
	(void)_args;
#endif

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
			CUBLAS_TRSM('R', 'U', 'N', 'U', ny21, nx21,
					(TYPE)1.0, sub11, ld11, sub21, ld21);

			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_CUBLAS_REPORT_ERROR(status);

			cudaStreamSynchronize(starpu_cuda_get_local_stream());

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}

#ifdef VERBOSE_KERNELS
	//fprintf(stderr, "OUTPUT 21 U11\n");
	fprintf(stderr, "OUTPUT 12 U11\n");
	STARPU_PLU(display_data_content)(sub11, nx21);
	//fprintf(stderr, "OUTPUT 21 U21\n");
	fprintf(stderr, "OUTPUT 12 U12\n");
	STARPU_PLU(display_data_content)(sub21, nx21);
#endif
}

static void STARPU_PLU(cpu_u21)(void *descr[], void *_args)
{
	STARPU_PLU(common_u21)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_u21)(void *descr[], void *_args)
{
	STARPU_PLU(common_u21)(descr, 1, _args);
}
#endif

static struct starpu_perfmodel STARPU_PLU(model_21) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_21_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_21_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_21_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_21)
#endif
};

struct starpu_codelet STARPU_PLU(cl21) =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {STARPU_PLU(cpu_u21)},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_u21)},
#endif
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &STARPU_PLU(model_21)
};


/*
 *	U11
 */

static inline void STARPU_PLU(common_u11)(void *descr[], int s, void *_args)
{
	TYPE *sub11;

	sub11 = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);

	unsigned long nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned long ld = STARPU_MATRIX_GET_LD(descr[0]);

	unsigned long z;

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
				STARPU_ASSERT(pivot != 0.0);

				CPU_SCAL(nx - z - 1, (1.0/pivot), &sub11[z+(z+1)*ld], ld);

				CPU_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				cudaMemcpyAsync(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
				cudaStreamSynchronize(starpu_cuda_get_local_stream());

				STARPU_ASSERT(pivot != 0.0);

				CUBLAS_SCAL(nx - z - 1, 1.0/pivot, &sub11[z+(z+1)*ld], ld);

				CUBLAS_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}

			cudaStreamSynchronize(starpu_cuda_get_local_stream());

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
#ifdef VERBOSE_KERNELS
	fprintf(stderr, "KERNEL 11 %d - k = %u\n", rank, info->k);
#endif
}

static void STARPU_PLU(cpu_u11)(void *descr[], void *_args)
{
	STARPU_PLU(common_u11)(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
static void STARPU_PLU(cublas_u11)(void *descr[], void *_args)
{
	STARPU_PLU(common_u11)(descr, 1, _args);
}
#endif// STARPU_USE_CUDA

static struct starpu_perfmodel STARPU_PLU(model_11) =
{
	.type = STARPU_HISTORY_BASED,
#ifdef STARPU_ATLAS
	.symbol = STARPU_PLU_STR(lu_model_11_atlas)
#elif defined(STARPU_GOTO)
	.symbol = STARPU_PLU_STR(lu_model_11_goto)
#elif defined(STARPU_OPENBLAS)
	.symbol = STARPU_PLU_STR(lu_model_11_openblas)
#else
	.symbol = STARPU_PLU_STR(lu_model_11)
#endif
};

struct starpu_codelet STARPU_PLU(cl11) =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {STARPU_PLU(cpu_u11)},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {STARPU_PLU(cublas_u11)},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &STARPU_PLU(model_11)
};
