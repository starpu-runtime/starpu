/*/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Télécom-SudParis
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


#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <pthread.h>
#include <signal.h>

#include <starpu.h>

static float *A, *B, *C;
static starpu_data_handle A_handle, B_handle, C_handle;

static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
static unsigned xdim = 1024;
static unsigned ydim = 1024;
static unsigned zdim = 512;


double mult_gemm_cost(starpu_buffer_descr *descr)
{
	/* C = A * B */
	uint32_t nxC, nyC, nxA;


	nxC = starpu_matrix_get_nx(descr[2].handle);
	nyC = starpu_matrix_get_ny(descr[2].handle);
	nxA = starpu_matrix_get_nx(descr[0].handle);

	//printf("nxC %d nxC %d nxA %d\n", nxC, nyC, nxA);

	double cost = ((double)nxC)*((double)nyC)*((double)nxA/1000.0f/4.11f);

	printf("cost %e \n", cost);

	return cost;
}

static void cpu_mult(void *descr[], __attribute__((unused))  void *arg)
{
	float *subA, *subB, *subC;
	uint32_t nxC, nyC, nyA;
	uint32_t ldA, ldB, ldC;
	printf("On application: Hello, this is kernel cpu_mult\n\n");
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

static void cpu_mult_2(void *descr[], __attribute__((unused))  void *arg)
{
	float *subA, *subB, *subC;
	uint32_t nxC, nyC, nyA;
	uint32_t ldA, ldB, ldC;
	printf("On application: this is kernel cpu_mult_2\n\n");
	/* .blas.ptr gives a pointer to the first element of the local copy */
	subA = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	subB = (float *)STARPU_MATRIX_GET_PTR(descr[1]);
	subC = (float *)STARPU_MATRIX_GET_PTR(descr[2]);

	nxC = STARPU_MATRIX_GET_NX(descr[2]);
	nyC = STARPU_MATRIX_GET_NY(descr[2]);
	nyA = STARPU_MATRIX_GET_NY(descr[0]);

	ldA = STARPU_MATRIX_GET_LD(descr[0]);
	ldB = STARPU_MATRIX_GET_LD(descr[1]);
	ldC = STARPU_MATRIX_GET_LD(descr[2]);

	/* we assume a FORTRAN-ordering! */
	unsigned i,j,k;
	for (j = 0; j < nxC; j++)
	{
		for (i = 0; i < nyC; i++)
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

	A = malloc(zdim*ydim*sizeof(float));
	B = malloc(xdim*zdim*sizeof(float));
	C = malloc(xdim*ydim*sizeof(float));

	/* fill the A and B matrices */
	srand(2009);
	for (j=0; j < ydim; j++) {
		for (i=0; i < zdim; i++) {
			A[j+i*ydim] = (float)(starpu_drand48());
		}
	}

	for (j=0; j < zdim; j++) {
		for (i=0; i < xdim; i++) {
			B[j+i*zdim] = (float)(starpu_drand48());
		}
	}

	for (j=0; j < ydim; j++) {
		for (i=0; i < xdim; i++) {
			C[j+i*ydim] = (float)(0);
		}
	}
}

static void partition_mult_data(void)
{
	/* note that we assume a FORTRAN ordering here! */

	starpu_matrix_data_register(&A_handle, 0, (uintptr_t)A,
		ydim, ydim, zdim, sizeof(float));
	starpu_matrix_data_register(&B_handle, 0, (uintptr_t)B,
		zdim, zdim, xdim, sizeof(float));
	starpu_matrix_data_register(&C_handle, 0, (uintptr_t)C,
		ydim, ydim, xdim, sizeof(float));

	/* A filter is a method to partition a data into disjoint chunks, it is
	 * described by the means of the "struct starpu_data_filter" structure that
	 * contains a function that is applied on a data handle to partition it
	 * into smaller chunks, and an argument that is passed to the function
	 * (eg. the number of blocks to create here).
	 */

	struct starpu_data_filter vert = {
		.filter_func = starpu_vertical_block_filter_func,
		.nchildren = nslicesx,
		.get_nchildren = NULL,
		.get_child_ops = NULL
	};

	struct starpu_data_filter horiz = {
		.filter_func = starpu_block_filter_func,
		.nchildren = nslicesy,
		.get_nchildren = NULL,
		.get_child_ops = NULL
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
 *	IMPORTANT: applying filters is equivalent to partitionning a piece of
 *	data in a hierarchical manner, so that memory consistency is enforced
 *	for each of the elements independantly. The tasks should therefore NOT
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

static struct starpu_perfmodel_t starpu_dgemm_model_common = {
	.cost_model = mult_gemm_cost,
	.type = STARPU_HISTORY_BASED,//STARPU_COMMON, //STARPU_PER_ARCH,
	.symbol = "mult_perf_model"
};

/*
static struct starpu_perfmodel_t mult_perf_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "mult_perf_model"
};
*/

struct starpu_conf conf = {
		.sched_policy_name = "heft",
		.calibrate = 1,
		.ncpus = 4
};


static starpu_codelet cl = {
        /* we can only execute that kernel on a CPU yet */
        .where = STARPU_CPU,
        //.starpu_impl_multiple = 1,
        /* CPU implementation of the codelet */
        .cpu_func = STARPU_MULTIPLE_CPU_IMPLEMENTATIONS,
        .cpu_funcs = {cpu_mult,cpu_mult_2},
        /* the codelet manipulates 3 buffers that are managed by the
         * DSM */
        .nbuffers = 3,
        /* in case the scheduling policy may use performance models */
        .model = &starpu_dgemm_model_common
};

static void launch_tasks(void)
{
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
			task->buffers[0].handle = starpu_data_get_sub_data(A_handle, 1, tasky);
			task->buffers[0].mode = STARPU_R;
			task->buffers[1].handle = starpu_data_get_sub_data(B_handle, 1, taskx);
			task->buffers[1].mode = STARPU_R;

			/* 2 filters were applied on matrix C, so we give
			 * starpu_data_get_sub_data 2 arguments. The order of the arguments
			 * must match the order in which the filters were
			 * applied.
			 * NB: starpu_data_get_sub_data(C_handle, 1, k) would have returned
			 * a handle to the column number k of matrix C.
			 * NB2: starpu_data_get_sub_data(C_handle, 2, taskx, tasky) is
			 * equivalent to
			 * starpu_data_get_sub_data(starpu_data_get_sub_data(C_handle, 1, taskx), 1, tasky)*/
			task->buffers[2].handle = starpu_data_get_sub_data(C_handle, 2, taskx, tasky);
			task->buffers[2].mode = STARPU_W;

			/* this is not a blocking call since task->synchronous = 0 */
			int summit_task;
			summit_task = starpu_task_submit(task);
			printf("task is submmited or not %d\n",summit_task);

		}
	}
}

int main(void)
{
	/* start the runtime */
	starpu_init(&conf);

	/* initialize matrices A, B and C and register them to StarPU */
	init_problem_data();

	/* partition matrices into blocks that can be manipulated by the
 	 * codelets */
	partition_mult_data();

	/* submit all tasks in an asynchronous fashion */
	launch_tasks();

	/* wait for termination */
	starpu_task_wait_for_all();

	/* remove the filters applied by the means of starpu_data_map_filters; now
 	 * it's not possible to manipulate a subset of C using starpu_data_get_sub_data until
	 * starpu_data_map_filters is called again on C_handle.
	 * The second argument is the memory node where the different subsets
	 * should be reassembled, 0 = main memory (RAM) */
	starpu_data_unpartition(C_handle, 0);

	/* stop monitoring matrix C : after this, it is not possible to pass C
	 * (or any subset of C) as a codelet input/output. This also implements
	 * a barrier so that the piece of data is put back into main memory in
	 * case it was only available on a GPU for instance. */
	starpu_data_unregister(C_handle);

	starpu_shutdown();

	return 0;
}
