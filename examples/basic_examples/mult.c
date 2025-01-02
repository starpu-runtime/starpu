/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010-2010  Mehdi Juhoor
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
 * This example shows a simple implementation of a blocked matrix
 * multiplication. Note that this is NOT intended to be an efficient
 * implementation of sgemm! In this example, we show:
 *  - how to declare dense matrices (starpu_matrix_data_register)
 *  - how to manipulate matrices within codelets (eg. descr[0].blas.ld)
 *  - how to use filters to partition the matrices into blocks
 *    (starpu_data_partition and starpu_data_map_filters)
 *  - how to unpartition data (starpu_data_unpartition) and how to stop
 *    monitoring data (starpu_data_unregister)
 *  - how to manipulate subsets of data (starpu_data_get_sub_data)
 *  - how to construct an autocalibrated performance model (starpu_perfmodel)
 *  - how to submit asynchronous tasks
 */

#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <signal.h>

#include <starpu.h>

static float *A, *B, *C, *Cref;
static starpu_data_handle_t A_handle, B_handle, C_handle;

static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
#ifdef STARPU_QUICK_CHECK
static unsigned xdim = 512;
static unsigned ydim = 512;
static unsigned zdim = 256;
#else
static unsigned xdim = 1024;
static unsigned ydim = 1024;
static unsigned zdim = 512;
#endif

extern void hip_mult(void *descr[], void *arg);
extern void cuda_mult(void *descr[], void *arg);

/*
 * That program should compute C = A * B
 *
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)
 *
 *             |---------------|
 *           z |       B       |
 *             |---------------|
 *      z              x
 *    |----|   |---------------|
 *    |    |   |               |
 *    |    |   |               |
 *    | A  | y |       C       |
 *    |    |   |               |
 *    |    |   |               |
 *    |----|   |---------------|
 *
 * Note: we use FORTRAN ordering.
 */

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

	/* ptr gives a pointer to the first element of the local copy */
	subA = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	subB = (float *)STARPU_MATRIX_GET_PTR(descr[1]);
	subC = (float *)STARPU_MATRIX_GET_PTR(descr[2]);

	/*
	 * Note: STARPU_MATRIX_GET_NX/NY is different from X/Y of the FORTRAN
	 * ordering:
	 * - nx is the number of consecutive elements (thus the number of rows
	 *   in FORTRAN order)
	 * - ny is the number of series that are separated by ld elements (thus
	 *   the number of columns in FORTRAN order)
	 * - ld stands for leading dimension
	 *
	 * NB: in case some filters were used, the leading dimension is not
	 * guaranteed to be the same in main memory (on the original matrix)
	 * and on the accelerator! */
	nxC = STARPU_MATRIX_GET_NX(descr[2]);
	nyC = STARPU_MATRIX_GET_NY(descr[2]);
	nyA = STARPU_MATRIX_GET_NY(descr[0]);

	ldA = STARPU_MATRIX_GET_LD(descr[0]);
	ldB = STARPU_MATRIX_GET_LD(descr[1]);
	ldC = STARPU_MATRIX_GET_LD(descr[2]);

	/* we use a FORTRAN-ordering! */
	unsigned i,j,k;
	for (i = 0; i < nyC; i++) /* iterate over columns of C */
	{
		for (j = 0; j < nxC; j++) /* iterate over rows of C */
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

static void init_problem_data(void)
{
	unsigned i,j;

	/* we initialize matrices A, B and C in the usual way */

	starpu_malloc_flags((void **)&A, zdim*ydim*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&B, xdim*zdim*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&C, xdim*ydim*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

	Cref = (float *) malloc(xdim*ydim*sizeof(float));
	assert(A);
	assert(B);
	assert(C);
	assert(Cref);

	/* fill the A and B matrices */
	starpu_srand48(2009);
	for (j=0; j < ydim; j++)
	{
		for (i=0; i < zdim; i++)
		{
			A[j+i*ydim] = (float)(starpu_drand48());
		}
	}

	for (j=0; j < zdim; j++)
	{
		for (i=0; i < xdim; i++)
		{
			B[j+i*zdim] = (float)(starpu_drand48());
		}
	}

	for (j=0; j < ydim; j++)
	{
		for (i=0; i < xdim; i++)
		{
			C[j+i*ydim] = (float)(0);
			Cref[j+i*ydim] = (float)(0);
		}
	}
}

static void partition_mult_data(void)
{
	/* note that we assume a FORTRAN ordering here! */

	/* The BLAS data interface is described by 4 parameters:
	 *  - the location of the first element of the matrix to monitor (3rd
	 *    argument)
	 *  - the number of elements between columns, aka leading dimension
	 *    (4th arg)
	 *  - the number of (contiguous) elements per column, ie. contiguous
	 *  elements (5th arg)
	 *  - the number of columns (6th arg)
	 * The first elements is a pointer to the data_handle that will be
	 * associated to the matrix, and the second elements gives the memory
	 * node in which resides the matrix: 0 means that the 3rd argument is
	 * an address in main memory.
	 */
	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A,
		ydim, ydim, zdim, sizeof(float));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B,
		zdim, zdim, xdim, sizeof(float));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C,
		ydim, ydim, xdim, sizeof(float));

	/* A filter is a method to partition a data into disjoint chunks, it is
	 * described by the means of the "struct starpu_data_filter" structure that
	 * contains a function that is applied on a data handle to partition it
	 * into smaller chunks, and an argument that is passed to the function
	 * (eg. the number of blocks to create here).
	 */

	/* StarPU supplies some basic filters such as the partition of a matrix
	 * into blocks, note that we are using a FORTRAN ordering so that the
	 * name of the filters are a bit misleading */
	struct starpu_data_filter vert =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nslicesx
	};

	struct starpu_data_filter horiz =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = nslicesy
	};

/*
 *	Illustration with nslicex = 4 and nslicey = 2, it is possible to access
 *	sub-data by using the "starpu_data_get_sub_data" method, which takes a data handle,
 *	the number of filters to apply, and the indexes for each filters, for
 *	instance:
 *
 *		A' handle is starpu_data_get_sub_data(A_handle, 1, 1);
 *		B' handle is starpu_data_get_sub_data(B_handle, 1, 2);
 *		C' handle is starpu_data_get_sub_data(C_handle, 2, 2, 1);
 *
 *	Note that here we applied 2 filters recursively onto C.
 *
 *	"starpu_data_get_sub_data(C_handle, 1, 3)" would return a handle to the 4th column
 *	of blocked matrix C for example.
 *
 *		              |---|---|---|---|
 *		              |   |   | B'|   | B
 *		              |---|---|---|---|
 *		                0   1   2   3
 *		     |----|   |---|---|---|---|
 *		     |    |   |   |   |   |   |
 *		     |    | 0 |   |   |   |   |
 *		     |----|   |---|---|---|---|
 *		     | A' |   |   |   | C'|   |
 *		     |    |   |   |   |   |   |
 *		     |----|   |---|---|---|---|
 *		       A              C
 *
 *	IMPORTANT: applying filters is equivalent to partitioning a piece of
 *	data in a hierarchical manner, so that memory consistency is enforced
 *	for each of the elements independently. The tasks should therefore NOT
 *	access inner nodes (eg. one column of C or the whole C) but only the
 *	leafs of the tree (ie. blocks here). Manipulating inner nodes is only
 *	possible by disapplying the filters (using starpu_data_unpartition), to
 *	enforce memory consistency.
 */

	starpu_data_partition(B_handle, &vert);
	starpu_data_partition(A_handle, &horiz);

	/* starpu_data_map_filters is a variable-arity function, the first argument
	 * is the handle of the data to partition, the second argument is the
	 * number of filters to apply recursively. Filters are applied in the
	 * same order as the arguments.
	 * This would be equivalent to starpu_data_partition(C_handle, &vert) and
	 * then applying horiz on each sub-data (ie. each column of C)
	 */
	starpu_data_map_filters(C_handle, 2, &vert, &horiz);
}

static struct starpu_perfmodel mult_perf_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "mult_perf_model"
};

static struct starpu_codelet cl =
{
	/* CPU implementation of the codelet */
	.cpu_funcs = {cpu_mult},
	.cpu_funcs_name = {"cpu_mult"},
#ifdef STARPU_USE_HIP
	/* HIP implementation of the codelet */
	.hip_funcs = {hip_mult},
	.hip_flags = {STARPU_HIP_ASYNC},
#endif
#ifdef STARPU_USE_CUDA
	/* CUDA implementation of the codelet */
	.cuda_funcs = {cuda_mult},
        .cuda_flags = {STARPU_CUDA_ASYNC},
	.where     = STARPU_CUDA,
#endif
	/* the codelet manipulates 3 buffers that are managed by the DSM */
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	/* in case the scheduling policy may use performance models */
	.model = &mult_perf_model
};

static int launch_tasks(void)
{
	int ret;
	/* partition the work into slices */
	unsigned taskx, tasky;

	for (taskx = 0; taskx < nslicesx; taskx++)
	{
		for (tasky = 0; tasky < nslicesy; tasky++)
		{
			/* C[taskx, tasky] = A[tasky] B[taskx] */

			/* by default, starpu_task_create() returns an
			 * asynchronous task (ie. task->synchronous = 0) */
			struct starpu_task *task = starpu_task_create();

			/* this task implements codelet "cl" */
			task->cl = &cl;

			/*
			 *              |---|---|---|---|
			 *              |   | * |   |   | B
			 *              |---|---|---|---|
			 *                    X
			 *     |----|   |---|---|---|---|
			 *     |****| Y |   |***|   |   |
			 *     |****|   |   |***|   |   |
			 *     |----|   |---|---|---|---|
			 *     |    |   |   |   |   |   |
			 *     |    |   |   |   |   |   |
			 *     |----|   |---|---|---|---|
			 *       A              C
			 */

			/* there was a single filter applied to matrices A
			 * (respectively B) so we grab the handle to the chunk
			 * identified by "tasky" (respectively "taskx). The "1"
			 * tells StarPU that there is a single argument to the
			 * variable-arity function starpu_data_get_sub_data */
			task->handles[0] = starpu_data_get_sub_data(A_handle, 1, tasky);
			task->handles[1] = starpu_data_get_sub_data(B_handle, 1, taskx);

			/* 2 filters were applied on matrix C, so we give
			 * starpu_data_get_sub_data 2 arguments. The order of the arguments
			 * must match the order in which the filters were
			 * applied.
			 * NB: starpu_data_get_sub_data(C_handle, 1, k) would have returned
			 * a handle to the column number k of matrix C.
			 * NB2: starpu_data_get_sub_data(C_handle, 2, taskx, tasky) is
			 * equivalent to
			 * starpu_data_get_sub_data(starpu_data_get_sub_data(C_handle, 1, taskx), 1, tasky)*/
			task->handles[2] = starpu_data_get_sub_data(C_handle, 2, taskx, tasky);

			/* this is not a blocking call since task->synchronous = 0 */
			ret = starpu_task_submit(task);
			if (ret == -ENODEV) return ret;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}
	return 0;
}

void check_result(float* C_gpu, float* C_ref, uint32_t ldC)
{
	unsigned i,j;
	for (i = 0; i < ydim; i++)
	{
		for (j = 0; j < xdim; j++)
		{
			if(C_gpu[j + i*ldC]-C_ref[j + i*ldC] > 1e-6*C_ref[j + i*ldC])
			{
				printf("| Cref[%u,%u]=%f - Cgpu[%u,%u]=%f | Error in the computation of C: the difference between the two is bigger than 1e-6 * the reference"
				       , i, j, C_ref[j + i*ldC], i, j, C_gpu[j + i*ldC]);
				exit(1);
			}
		}
	}
	printf("SUCCESSFUL COMPUTATION\n");
}

int main(void)
{
	int ret;

	/* start the runtime */
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* initialize matrices A, B and C and register them to StarPU */
	init_problem_data();

	/* partition matrices into blocks that can be manipulated by the
	 * codelets */
	partition_mult_data();

	/* submit all tasks in an asynchronous fashion */
	ret = launch_tasks();
	if (ret == -ENODEV) goto enodev;

	/* cpu compution to check */
	/* ============================================= */

	uint32_t ldA = ydim;
	uint32_t ldB = zdim;
	uint32_t ldC = ydim;

	unsigned i,j,k;
	for (i = 0; i < ydim; i++)
	{
		for (j = 0; j < xdim; j++)
		{
			float sum = 0.0;

			for (k = 0; k < zdim; k++)
			{
				sum += A[j+k*ldA]*B[k+i*ldB];
			}

			Cref[j + i*ldC] = sum;
		}
	}

	/* ============================================= */
	/* wait for termination */
	starpu_task_wait_for_all();

	/* remove the filters applied by the means of starpu_data_map_filters; now
	 * it's not possible to manipulate a subset of C using starpu_data_get_sub_data until
	 * starpu_data_map_filters is called again on C_handle.
	 * The second argument is the memory node where the different subsets
	 * should be reassembled, 0 = main memory (RAM) */
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);

	/* stop monitoring matrix C : after this, it is not possible to pass C
	 * (or any subset of C) as a codelet input/output. This also implements
	 * a barrier so that the piece of data is put back into main memory in
	 * case it was only available on a GPU for instance. */
	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

	/* Comment to remove printing of results */
	check_result(C, Cref, ldC);

        starpu_free_flags(A, zdim*ydim*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(B, xdim*zdim*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(C, xdim*ydim*sizeof(float), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

	free(Cref);

	starpu_shutdown();

	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
