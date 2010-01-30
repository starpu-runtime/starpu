/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include "pxlu.h"
#include "pxlu_kernels.h"
#include <math.h>

/*
 *   U22 
 */

static inline void STARPU_PLU(common_u22)(void *descr[],
				int s, __attribute__((unused)) void *_args)
{
	TYPE *right 	= (TYPE *)GET_BLAS_PTR(descr[0]);
	TYPE *left 	= (TYPE *)GET_BLAS_PTR(descr[1]);
	TYPE *center 	= (TYPE *)GET_BLAS_PTR(descr[2]);

	unsigned dx = GET_BLAS_NX(descr[2]);
	unsigned dy = GET_BLAS_NY(descr[2]);
	unsigned dz = GET_BLAS_NY(descr[0]);

	unsigned ld12 = GET_BLAS_LD(descr[0]);
	unsigned ld21 = GET_BLAS_LD(descr[1]);
	unsigned ld22 = GET_BLAS_LD(descr[2]);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//fprintf(stderr, "KERNEL 22 %d\n", rank);

#ifdef USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif

	switch (s) {
		case 0:
			CPU_GEMM("N", "N", dy, dx, dz, 
				(TYPE)-1.0, right, ld21, left, ld12,
				(TYPE)1.0, center, ld22);
			break;

#ifdef USE_CUDA
		case 1:
			CUBLAS_GEMM('n', 'n', dx, dy, dz,
				(TYPE)-1.0, right, ld21, left, ld12,
				(TYPE)1.0f, center, ld22);

			status = cublasGetError();
			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_ABORT();

			if (STARPU_UNLIKELY((cures = cudaThreadSynchronize()) != cudaSuccess))
				CUDA_REPORT_ERROR(cures);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

static void STARPU_PLU(cpu_u22)(void *descr[], void *_args)
{
	STARPU_PLU(common_u22)(descr, 0, _args);
}

#ifdef USE_CUDA
static void STARPU_PLU(cublas_u22)(void *descr[], void *_args)
{
	STARPU_PLU(common_u22)(descr, 1, _args);
}
#endif// USE_CUDA

static struct starpu_perfmodel_t STARPU_PLU(model_22) = {
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = STARPU_PLU_STR(lu_model_22_atlas)
#elif defined(GOTO)
	.symbol = STARPU_PLU_STR(lu_model_22_goto)
#else
	.symbol = STARPU_PLU_STR(lu_model_22)
#endif
};

starpu_codelet STARPU_PLU(cl22) = {
	.where = CORE|CUDA,
	.core_func = STARPU_PLU(cpu_u22),
#ifdef USE_CUDA
	.cuda_func = STARPU_PLU(cublas_u22),
#endif
	.nbuffers = 3,
	.model = &STARPU_PLU(model_22)
};


/*
 * U12
 */

static inline void STARPU_PLU(common_u12)(void *descr[],
				int s, __attribute__((unused)) void *_args)
{
	TYPE *sub11;
	TYPE *sub12;

	sub11 = (TYPE *)GET_BLAS_PTR(descr[0]);	
	sub12 = (TYPE *)GET_BLAS_PTR(descr[1]);

	unsigned ld11 = GET_BLAS_LD(descr[0]);
	unsigned ld12 = GET_BLAS_LD(descr[1]);

	unsigned nx12 = GET_BLAS_NX(descr[1]);
	unsigned ny12 = GET_BLAS_NY(descr[1]);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	fprintf(stderr, "KERNEL 12 %d\n", rank);

#ifdef USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif

//	fprintf(stderr, "INPUT 12 U11\n");
//	STARPU_PLU(display_data_content)(sub11, nx12);
//	fprintf(stderr, "INPUT 12 U12\n");
//	STARPU_PLU(display_data_content)(sub12, nx12);



	/* solve L11 U12 = A12 (find U12) */
	switch (s) {
		case 0:
			CPU_TRSM("L", "L", "N", "N", nx12, ny12,
					(TYPE)1.0, sub11, ld11, sub12, ld12);
			break;
#ifdef USE_CUDA
		case 1:
			CUBLAS_TRSM('L', 'L', 'N', 'N', ny12, nx12,
					(TYPE)1.0, sub11, ld11, sub12, ld12);

			status = cublasGetError();
			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_ABORT();

			if (STARPU_UNLIKELY((cures = cudaThreadSynchronize()) != cudaSuccess))
				CUDA_REPORT_ERROR(cures);

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}

//	fprintf(stderr, "OUTPUT 12 U12\n");
//	STARPU_PLU(display_data_content)(sub12, nx12);
}

static void STARPU_PLU(cpu_u12)(void *descr[], void *_args)
{
	STARPU_PLU(common_u12)(descr, 0, _args);
}

#ifdef USE_CUDA
static void STARPU_PLU(cublas_u12)(void *descr[], void *_args)
{
	STARPU_PLU(common_u12)(descr, 1, _args);
}
#endif // USE_CUDA

static struct starpu_perfmodel_t STARPU_PLU(model_12) = {
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = STARPU_PLU_STR(lu_model_12_atlas)
#elif defined(GOTO)
	.symbol = STARPU_PLU_STR(lu_model_12_goto)
#else
	.symbol = STARPU_PLU_STR(lu_model_12)
#endif
};

starpu_codelet STARPU_PLU(cl12) = {
	.where = CORE|CUDA,
	.core_func = STARPU_PLU(cpu_u12),
#ifdef USE_CUDA
	.cuda_func = STARPU_PLU(cublas_u12),
#endif
	.nbuffers = 2,
	.model = &STARPU_PLU(model_12)
};


/* 
 * U21
 */

static inline void STARPU_PLU(common_u21)(void *descr[],
				int s, __attribute__((unused)) void *_args)
{
	TYPE *sub11;
	TYPE *sub21;

	sub11 = (TYPE *)GET_BLAS_PTR(descr[0]);
	sub21 = (TYPE *)GET_BLAS_PTR(descr[1]);

	unsigned ld11 = GET_BLAS_LD(descr[0]);
	unsigned ld21 = GET_BLAS_LD(descr[1]);

	unsigned nx21 = GET_BLAS_NX(descr[1]);
	unsigned ny21 = GET_BLAS_NY(descr[1]);
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	fprintf(stderr, "KERNEL 21 %d \n", rank);

	//fprintf(stderr, "INPUT 21 U11\n");
	//STARPU_PLU(display_data_content)(sub11, nx21);
	//fprintf(stderr, "INPUT 21 U12\n");
	//STARPU_PLU(display_data_content)(sub21, nx21);



#ifdef USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif


	switch (s) {
		case 0:
			CPU_TRSM("R", "U", "N", "U", nx21, ny21,
					(TYPE)1.0, sub11, ld11, sub21, ld21);
			break;
#ifdef USE_CUDA
		case 1:
			CUBLAS_TRSM('R', 'U', 'N', 'U', ny21, nx21,
					(TYPE)1.0, sub11, ld11, sub21, ld21);

			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_ABORT();

			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}

//	fprintf(stderr, "INPUT 21 U21\n");
//	STARPU_PLU(display_data_content)(sub21, nx21);


}

static void STARPU_PLU(cpu_u21)(void *descr[], void *_args)
{
	STARPU_PLU(common_u21)(descr, 0, _args);
}

#ifdef USE_CUDA
static void STARPU_PLU(cublas_u21)(void *descr[], void *_args)
{
	STARPU_PLU(common_u21)(descr, 1, _args);
}
#endif 

static struct starpu_perfmodel_t STARPU_PLU(model_21) = {
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = STARPU_PLU_STR(lu_model_21_atlas)
#elif defined(GOTO)
	.symbol = STARPU_PLU_STR(lu_model_21_goto)
#else
	.symbol = STARPU_PLU_STR(lu_model_21)
#endif
};

starpu_codelet STARPU_PLU(cl21) = {
	.where = CORE|CUDA,
	.core_func = STARPU_PLU(cpu_u21),
#ifdef USE_CUDA
	.cuda_func = STARPU_PLU(cublas_u21),
#endif
	.nbuffers = 2,
	.model = &STARPU_PLU(model_21)
};


/*
 *	U11
 */

static inline void STARPU_PLU(common_u11)(void *descr[],
				int s, __attribute__((unused)) void *_args)
{
	TYPE *sub11;

	sub11 = (TYPE *)GET_BLAS_PTR(descr[0]); 

	unsigned long nx = GET_BLAS_NX(descr[0]);
	unsigned long ld = GET_BLAS_LD(descr[0]);

	unsigned long z;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	fprintf(stderr, "KERNEL 11 %d\n", rank);

	switch (s) {
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
#ifdef USE_CUDA
		case 1:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				cudaMemcpy(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost);
				cudaStreamSynchronize(0);

				STARPU_ASSERT(pivot != 0.0);
				
				CUBLAS_SCAL(nx - z - 1, 1.0/pivot, &sub11[z+(z+1)*ld], ld);
				
				CUBLAS_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}
			
			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

static void STARPU_PLU(cpu_u11)(void *descr[], void *_args)
{
	STARPU_PLU(common_u11)(descr, 0, _args);
}

#ifdef USE_CUDA
static void STARPU_PLU(cublas_u11)(void *descr[], void *_args)
{
	STARPU_PLU(common_u11)(descr, 1, _args);
}
#endif// USE_CUDA

static struct starpu_perfmodel_t STARPU_PLU(model_11) = {
	.type = HISTORY_BASED,
#ifdef ATLAS
	.symbol = STARPU_PLU_STR(lu_model_11_atlas)
#elif defined(GOTO)
	.symbol = STARPU_PLU_STR(lu_model_11_goto)
#else
	.symbol = STARPU_PLU_STR(lu_model_11)
#endif
};

starpu_codelet STARPU_PLU(cl11) = {
	.where = CORE|CUDA,
	.core_func = STARPU_PLU(cpu_u11),
#ifdef USE_CUDA
	.cuda_func = STARPU_PLU(cublas_u11),
#endif
	.nbuffers = 1,
	.model = &STARPU_PLU(model_11)
};


