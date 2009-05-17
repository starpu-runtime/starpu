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

void callback_func(void *arg)
{
	/* the argument is a pointer to a counter of the remaining tasks */
	int *counterptr = arg;

	int counter = STARPU_ATOMIC_ADD(counterptr, -1);
	if (counter == 0)
	{
		/* we are done */	
		pthread_mutex_lock(&mutex);
		terminated = 1;
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
	}
}



void cpu_mult(starpu_data_interface_t *descr, __attribute__((unused))  void *arg)
{
	float *subA, *subB, *subC;
	uint32_t nxC, nyC, nyA;
	uint32_t ldA, ldB, ldC;

	subA = (float *)descr[0].blas.ptr;
	subB = (float *)descr[1].blas.ptr;
	subC = (float *)descr[2].blas.ptr;

	nxC = descr[2].blas.nx;
	nyC = descr[2].blas.ny;
	nyA = descr[0].blas.ny;

	ldA = descr[0].blas.ld;
	ldB = descr[1].blas.ld;
	ldC = descr[2].blas.ld;

	/* we assume a FORTRAN-ordering ! */
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

			subC[j + i*ldC] += sum;
		}
	}
}

static void init_problem_data(void)
{
	unsigned i,j;

	A = malloc(zdim*ydim*sizeof(float));
	B = malloc(xdim*zdim*sizeof(float));
	C = malloc(xdim*ydim*sizeof(float));

	/* fill the A and B matrices */
	srand(2008);
	for (j=0; j < ydim; j++) {
		for (i=0; i < zdim; i++) {
			A[j+i*ydim] = (float)(drand48());
		}
	}

	for (j=0; j < zdim; j++) {
		for (i=0; i < xdim; i++) {
			B[j+i*zdim] = (float)(drand48());
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
	starpu_register_blas_data(&A_handle, 0, (uintptr_t)A, 
		ydim, ydim, zdim, sizeof(float));
	starpu_register_blas_data(&B_handle, 0, (uintptr_t)B, 
		zdim, zdim, xdim, sizeof(float));
	starpu_register_blas_data(&C_handle, 0, (uintptr_t)C, 
		ydim, ydim, xdim, sizeof(float));

	starpu_filter f;
	f.filter_func = starpu_vertical_block_filter_func;
	f.filter_arg = nslicesx;
		
	starpu_filter f2;
	f2.filter_func = starpu_block_filter_func;
	f2.filter_arg = nslicesy;
		
	starpu_partition_data(B_handle, &f);
	starpu_partition_data(A_handle, &f2);

	starpu_map_filters(C_handle, 2, &f, &f2);
}

static struct starpu_perfmodel_t mult_perf_model = {
	.type = HISTORY_BASED,
	.symbol = "mult_perf_model"
};

static void launch_tasks(void)
{
	/* partition the work into slices */
	unsigned taskx, tasky;

	taskcounter = nslicesx * nslicesy;

	starpu_codelet cl = {
		.where = CORE,
		.core_func = cpu_mult,
		.nbuffers = 3,
		.model = &mult_perf_model
	};


	for (taskx = 0; taskx < nslicesx; taskx++) 
	{
		for (tasky = 0; tasky < nslicesy; tasky++)
		{
			/* A B[task] = C[task] */
			struct starpu_task *task = starpu_task_create();

			task->cl = &cl;

			task->callback_func = callback_func;
			task->callback_arg = &taskcounter;

			task->buffers[0].handle = get_sub_data(A_handle, 1, tasky);
			task->buffers[0].mode = STARPU_R;
			task->buffers[1].handle = get_sub_data(B_handle, 1, taskx);
			task->buffers[1].mode = STARPU_R;
			task->buffers[2].handle = 
				get_sub_data(C_handle, 2, taskx, tasky);
			task->buffers[2].mode = STARPU_RW;

			starpu_submit_task(task);
		}
	}
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	/* start the runtime */
	starpu_init(NULL);

	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);

	init_problem_data();

	partition_mult_data();

	launch_tasks();

	pthread_mutex_lock(&mutex);
	if (!terminated)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	starpu_unpartition_data(C_handle, 0);
	starpu_delete_data(C_handle);
	
	starpu_shutdown();

	return 0;
}
