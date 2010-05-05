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

/*
 * This example shows a simple implementation of a blocked matrix
 * multiplication. Note that this is NOT intended to be an efficient
 * implementation of sgemm! In this example, we show:
 *  - how to declare dense matrices (starpu_matrix_data_register)
 *  - how to manipulate matrices within codelets (eg. descr[0].blas.ld)
 *  - how to use filters to partition the matrices into blocks
 *    (starpu_data_partition and starpu_map_filters)
 *  - how to unpartition data (starpu_data_unpartition) and how to stop
 *    monitoring data (starpu_data_unregister)
 *  - how to manipulate subsets of data (starpu_data_get_sub_data)
 *  - how to construct an autocalibrated performance model (starpu_perfmodel_t)
 *  - how to submit asynchronous tasks and how to use callback to handle task
 *    termination
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

static pthread_mutex_t mutex;
static pthread_cond_t cond;
static unsigned taskcounter;
static unsigned terminated = 0;

static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
static unsigned nslicesz = 4;
static unsigned xdim = 1024;
static unsigned ydim = 1024;
static unsigned zdim = 512;


/*
 * That program should compute C = A * B 
 * 
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)

              |---------------|
            z |       B       |
              |---------------|
       z              x
     |----|   |---------------|
     |    |   |               |
     |    |   |               |
     | A  | y |       C       |
     |    |   |               |
     |    |   |               |
     |----|   |---------------|

 */

static void callback_func(void *arg)
{
	/* the argument is a pointer to a counter of the remaining tasks */
	int *counterptr = arg;

	/* counterptr points to a variable with the number of remaining tasks,
 	 * when it reaches 0, all tasks are done */
	int counter = STARPU_ATOMIC_ADD(counterptr, -1);
	if (counter == 0)
	{
		/* IMPORTANT : note that we CANNOT call blocking operations
		 * within callbacks as it may lead to a deadlock of StarPU.
		 * starpu_data_unpartition is for instance called by the main
		 * thread since it may cause /potentially/ blocking operations
		 * such as memory transfers from a GPU to a CPU. */
		
		/* wake the application to notify the termination of all the
 		 * tasks */
		pthread_mutex_lock(&mutex);
		terminated = 1;
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
	}
}

/*
 * The codelet is passed 3 matrices, the "descr" union-type field gives a
 * description of the layout of those 3 matrices in the local memory (ie. RAM
 * in the case of CPU, GPU frame buffer in the case of GPU etc.). Since we have
 * registered data with the "blas" data interface, we manipulate the .blas
 * field of the descr[x] elements which are union types.
 */

static void cpu_mult(void *descr[], __attribute__((unused))  void *arg)
{
	float *subA, *subB, *subC;
	uint32_t nxC, nyC, nyA;
	uint32_t ldA, ldB, ldC;

	/* .blas.ptr gives a pointer to the first element of the local copy */
	subA = (float *)STARPU_GET_MATRIX_PTR(descr[0]);
	subB = (float *)STARPU_GET_MATRIX_PTR(descr[1]);
	subC = (float *)STARPU_GET_MATRIX_PTR(descr[2]);

	/* .blas.nx is the number of rows (consecutive elements) and .blas.ny
	 * is the number of lines that are separated by .blas.ld elements (ld
	 * stands for leading dimension).
	 * NB: in case some filters were used, the leading dimension is not
	 * guaranteed to be the same in main memory (on the original matrix)
	 * and on the accelerator! */
	nxC = STARPU_GET_MATRIX_NX(descr[2]);
	nyC = STARPU_GET_MATRIX_NY(descr[2]);
	nyA = STARPU_GET_MATRIX_NY(descr[0]);

	ldA = STARPU_GET_MATRIX_LD(descr[0]);
	ldB = STARPU_GET_MATRIX_LD(descr[1]);
	ldC = STARPU_GET_MATRIX_LD(descr[2]);

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

	/* The BLAS data interface is described by 4 parameters: 
	 *  - the location of the first element of the matrix to monitor (3rd
	 *    argument)
	 *  - the number of elements per row, ie. contiguous elements (4th arg)
	 *  - the number of elements between columns, aka leading dimension
	 *    (5th arg)
	 *  - the number of columns (6th arg)
	 * The first elements is a pointer to the data_handle that will be
	 * associated to the matrix, and the second elements gives the memory
	 * node in which resides the matrix: 0 means that the 3rd argument is
	 * an adress in main memory.
	 */
	starpu_matrix_data_register(&A_handle, 0, (uintptr_t)A, 
		ydim, ydim, zdim, sizeof(float));
	starpu_matrix_data_register(&B_handle, 0, (uintptr_t)B, 
		zdim, zdim, xdim, sizeof(float));
	starpu_matrix_data_register(&C_handle, 0, (uintptr_t)C, 
		ydim, ydim, xdim, sizeof(float));

	/* A filter is a method to partition a data into disjoint chunks, it is
	 * described by the means of the "starpu_filter" structure that
	 * contains a function that is applied on a data handle to partition it
	 * into smaller chunks, and an argument that is passed to the function
	 * (eg. the number of blocks to create here).
	 */

	/* StarPU supplies some basic filters such as the partition of a matrix
	 * into blocks, note that we are using a FORTRAN ordering so that the
	 * name of the filters are a bit misleading */
	starpu_filter f = {
		.filter_func = starpu_vertical_block_filter_func,
		.filter_arg = nslicesx
	};
		
	starpu_filter f2 = {
		.filter_func = starpu_block_filter_func,
		.filter_arg = nslicesy
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

	starpu_data_partition(B_handle, &f);
	starpu_data_partition(A_handle, &f2);

	/* starpu_map_filters is a variable-arity function, the first argument
	 * is the handle of the data to partition, the second argument is the
	 * number of filters to apply recursively. Filters are applied in the
	 * same order as the arguments.
	 * This would be equivalent to starpu_data_partition(C_handle, &f) and
	 * then applying f2 on each sub-data (ie. each column of C)
	 */
	starpu_map_filters(C_handle, 2, &f, &f2);
}

static struct starpu_perfmodel_t mult_perf_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "mult_perf_model"
};

static void launch_tasks(void)
{
	/* partition the work into slices */
	unsigned taskx, tasky;

	/* the callback decrements this value every time a task is terminated
	 * and notify the termination of the computation to the application
	 * when the counter reaches 0 */
	taskcounter = nslicesx * nslicesy;

	starpu_codelet cl = {
		/* we can only execute that kernel on a CPU yet */
		.where = STARPU_CPU,
		/* CPU implementation of the codelet */
		.cpu_func = cpu_mult,
		/* the codelet manipulates 3 buffers that are managed by the
 		 * DSM */
		.nbuffers = 3,
		/* in case the scheduling policy may use performance models */
		.model = &mult_perf_model
	};

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

			task->callback_func = callback_func;
			task->callback_arg = &taskcounter;

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
			starpu_task_submit(task);
		}
	}
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);

	/* start the runtime */
	starpu_init(NULL);

	/* initialize matrices A, B and C and register them to StarPU */
	init_problem_data();

	/* partition matrices into blocks that can be manipulated by the
 	 * codelets */
	partition_mult_data();

	/* submit all tasks in an asynchronous fashion */
	launch_tasks();

	/* the different tasks are asynchronous so we use a callback to get
	 * notified of the termination of the computation */
	pthread_mutex_lock(&mutex);
	if (!terminated)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	/* remove the filters applied by the means of starpu_map_filters; now
 	 * it's not possible to manipulate a subset of C using starpu_data_get_sub_data until
	 * starpu_map_filters is called again on C_handle.
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
