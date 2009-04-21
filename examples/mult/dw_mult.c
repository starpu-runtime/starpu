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

#include "dw_mult.h"

#define TAG(taskx, tasky)	((((unsigned long long)(taskx))<<32) | (unsigned long long)(tasky))



float *A, *B, *C;
starpu_data_handle A_handle, B_handle, C_handle;

pthread_mutex_t mutex;
pthread_cond_t cond;

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

void terminate(void)
{

	fprintf(stderr, "unpartition !!\n");
	starpu_unpartition_data(C_handle, 0);

	starpu_delete_data(C_handle);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	uint64_t total_flop = BLAS3_FLOP(ydim, xdim, zdim);
	uint64_t total_ls = ls_cublas + ls_atlas;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);
	fprintf(stderr, "	GB : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_ls/1000000000.0f, (double)ls_cublas/1000000000.0f, (double)ls_atlas/1000000000.0f);
	fprintf(stderr, "	GB/s : %2.2f\n", (double)total_ls / (double)timing/1000);

#ifdef CHECK_OUTPUT
	/* check results */
	/* compute C = C - AB */

	SGEMM("N", "N", ydim, xdim, zdim, -1.0f, A, ydim, B, zdim, 1.0f, C, ydim);
		
	/* make sure C = 0 */
	float err;
	err = SASUM(xdim*ydim, C, 1);	
	
	if (err < xdim*ydim*0.001) {
		fprintf(stderr, "Results are OK\n");
	}
	else {
		fprintf(stderr, "There were errors ... err = %f\n", err);
	}
#endif // CHECK_OUTPUT

	pthread_mutex_lock(&mutex);
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);
}

void callback_func(void *arg)
{
	/* the argument is a pointer to a counter of the remaining tasks */
	int *counterptr = arg;

	int counter = STARPU_ATOMIC_ADD(counterptr, -1);
	if (counter == 0)
	{
		/* we are done */	
		fprintf(stderr, "done ...\n");
		terminate();
	}

	return;
}


#define COMMON_CODE			\
	uint32_t nxC, nyC, nyA;		\
	uint32_t ldA, ldB, ldC;		\
					\
	float *subA;			\
	float *subB;			\
	float *subC;			\
					\
	subA = (float *)descr[0].blas.ptr;	\
	subB = (float *)descr[1].blas.ptr;	\
	subC = (float *)descr[2].blas.ptr;	\
					\
	nxC = descr[2].blas.nx;		\
	nyC = descr[2].blas.ny;		\
	nyA = descr[0].blas.ny;		\
					\
	ldA = descr[0].blas.ld;		\
	ldB = descr[1].blas.ld;		\
	ldC = descr[2].blas.ld;



#ifdef USE_CUDA
void cublas_mult(starpu_data_interface_t *descr, __attribute__((unused)) void *arg)
{
	COMMON_CODE

	cublasSgemm('n', 'n', nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 
					     0.0f, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		STARPU_ASSERT(0);

	uint64_t flopcnt = BLAS3_FLOP(nyC, nxC, nyA);

	flop_cublas += flopcnt;
	ls_cublas += BLAS3_LS(nyC, nxC, nyA);
}
#endif

void core_mult(starpu_data_interface_t *descr, __attribute__((unused))  void *arg)
{
	COMMON_CODE

	SGEMM("N", "N", nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 0.0f, subC, ldC);

	flop_atlas += BLAS3_FLOP(nxC, nyC, nyA);
	ls_atlas += BLAS3_LS(nxC, nyC, nyA);
}

static void init_problem_data(void)
{
	unsigned i,j;

#ifdef USE_CUDA
	if (pin) {
		starpu_malloc_pinned_if_possible(&A, zdim*ydim*sizeof(float));
		starpu_malloc_pinned_if_possible(&B, xdim*zdim*sizeof(float));
		starpu_malloc_pinned_if_possible(&C, xdim*ydim*sizeof(float));
	} else
#endif
	{
#ifdef HAVE_POSIX_MEMALIGN
		posix_memalign((void **)&A, 4096, zdim*ydim*sizeof(float));
		posix_memalign((void **)&B, 4096, xdim*zdim*sizeof(float));
		posix_memalign((void **)&C, 4096, xdim*ydim*sizeof(float));
#else
		A = malloc(zdim*ydim*sizeof(float));
		B = malloc(xdim*zdim*sizeof(float));
		C = malloc(xdim*ydim*sizeof(float));
#endif
	}

	/* fill the A and B matrices */
	if (norandom) {
		for (j=0; j < ydim; j++) {
			for (i=0; i < zdim; i++) {
				A[j+i*ydim] = (float)(i);
			}
		}
	
		for (j=0; j < zdim; j++) {
			for (i=0; i < xdim; i++) {
				B[j+i*zdim] = (float)(j);
			}
		}
	} 
	else {
#ifdef NORANDOM
		srand(2008);
		STARPU_ASSERT(0);
#endif
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
	}

	for (j=0; j < ydim; j++) {
		for (i=0; i < xdim; i++) {
			C[j+i*ydim] = (float)(0);
		}
	}

	display_memory_consumption();
}

static void partition_mult_data(void)
{
	gettimeofday(&start, NULL);

	starpu_monitor_blas_data(&A_handle, 0, (uintptr_t)A, 
		ydim, ydim, zdim, sizeof(float));
	starpu_monitor_blas_data(&B_handle, 0, (uintptr_t)B, 
		zdim, zdim, xdim, sizeof(float));
	starpu_monitor_blas_data(&C_handle, 0, (uintptr_t)C, 
		ydim, ydim, xdim, sizeof(float));

	conf.k = zdim;
	conf.m = ydim/nslicesy;
	conf.n = xdim/nslicesx;

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

static void launch_codelets(void)
{
#ifdef USE_FXT
	fxt_register_thread(0);
#endif
	/* partition the work into slices */
	unsigned taskx, tasky;

	taskcounter = nslicesx * nslicesy;

	srand(time(NULL));

	starpu_codelet cl = {
		.where = CORE|CUBLAS|GORDON,
		.core_func = core_mult,
#ifdef USE_CUDA
		.cublas_func = cublas_mult,
#endif
#ifdef USE_GORDON
#ifdef SPU_FUNC_SGEMM
		.gordon_func = SPU_FUNC_SGEMM,
#else
#warning SPU_FUNC_SGEMM is not available
#endif
#endif
		.nbuffers = 3
	};

	/* should we use a single performance model for all archs and use an
 	 * acceleration factor ? */
	if (use_common_model) {
		cl.model = &sgemm_model_common;
	}
	else {
		cl.model = &sgemm_model;
	}

	for (taskx = 0; taskx < nslicesx; taskx++) 
	{
		for (tasky = 0; tasky < nslicesy; tasky++)
		{
			/* A B[task] = C[task] */
			struct starpu_task *task = starpu_task_create();

			task->cl = &cl;
			task->cl_arg = &conf;
			task->cl_arg_size = sizeof(struct block_conf);

			task->callback_func = callback_func;
			task->callback_arg = &taskcounter;

			starpu_tag_t tag = TAG(taskx, tasky); 

			task->use_tag = 1;
			task->tag_id = tag;

			task->buffers[0].state = get_sub_data(A_handle, 1, tasky);
			task->buffers[0].mode = STARPU_R;
			task->buffers[1].state = get_sub_data(B_handle, 1, taskx);
			task->buffers[1].mode = STARPU_R;
			task->buffers[2].state = 
				get_sub_data(C_handle, 2, taskx, tasky);
			task->buffers[2].mode = STARPU_RW;

			starpu_submit_task(task);
		}
	}
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	starpu_init(NULL);

	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);

	init_problem_data();

	partition_mult_data();

	launch_codelets();

	pthread_mutex_lock(&mutex);
	pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	starpu_shutdown();

	return 0;
}
