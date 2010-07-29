/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

#include "dw_factolu.h"
#include <sys/time.h>

uint8_t *advance_12_21; /* size nblocks*nblocks */
uint8_t *advance_11; /* size nblocks*nblocks */
uint8_t *advance_22; /* array of nblocks *nblocks*nblocks */

struct timeval start;
struct timeval end;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static unsigned finished = 0;

static unsigned no_prio = 0;

static starpu_codelet cl11 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dw_cpu_codelet_update_u11,
#ifdef STARPU_USE_CUDA
	.cuda_func = dw_cublas_codelet_update_u11,
#endif
	.nbuffers = 1,
	.model = &model_11
};

static starpu_codelet cl12 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dw_cpu_codelet_update_u12,
#ifdef STARPU_USE_CUDA
	.cuda_func = dw_cublas_codelet_update_u12,
#endif
	.nbuffers = 2,
	.model = &model_12
}; 

static starpu_codelet cl21 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dw_cpu_codelet_update_u21,
#ifdef STARPU_USE_CUDA
	.cuda_func = dw_cublas_codelet_update_u21,
#endif
	.nbuffers = 2,
	.model = &model_21
}; 

static starpu_codelet cl22 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dw_cpu_codelet_update_u22,
#ifdef STARPU_USE_CUDA
	.cuda_func = dw_cublas_codelet_update_u22,
#endif
	.nbuffers = 3,
	.model = &model_22
}; 



#define STARTED	0x01
#define DONE	0x10

/*
 *	Upgraded Callbacks : break the pipeline design !
 */

void dw_callback_v2_codelet_update_u22(void *argcb)
{
	cl_args *args = argcb;	

	unsigned k = args->k;
	unsigned i = args->i;
	unsigned j = args->j;
	unsigned nblocks = args->nblocks;

	/* we did task 22k,i,j */
	advance_22[k*nblocks*nblocks + i + j*nblocks] = DONE;
	
	if ( (i == j) && (i == k+1)) {
		/* we now reduce the LU22 part (recursion appears there) */
		cl_args *u11arg = malloc(sizeof(cl_args));

		struct starpu_task *task = starpu_task_create();
			task->callback_func = dw_callback_v2_codelet_update_u11;
			task->callback_arg = u11arg;
			task->cl = &cl11;
			task->cl_arg = u11arg;

			task->buffers[0].handle =
				starpu_data_get_sub_data(args->dataA, 2, k+1, k+1);
			task->buffers[0].mode = STARPU_RW;
	
		u11arg->dataA = args->dataA;
		u11arg->i = k + 1;
		u11arg->nblocks = args->nblocks;

		/* schedule the codelet */
		if (!no_prio)
			task->priority = STARPU_MAX_PRIO;

		starpu_task_submit(task);
	}

	/* 11k+1 + 22k,k+1,j => 21 k+1,j */
	if ( i == k + 1) {
		uint8_t dep;
		/* 11 k+1*/
		dep = advance_11[(k+1)];
		if (dep & DONE) {
			/* try to push the task */
			uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[(k+1) + j*nblocks], STARTED);
				if ((u & STARTED) == 0) {
					/* we are the only one that should 
					 * launch that task */
					cl_args *u21a = malloc(sizeof(cl_args));

					struct starpu_task *task21 = starpu_task_create();
					task21->callback_func = dw_callback_v2_codelet_update_u21;
					task21->callback_arg = u21a;
					task21->cl = &cl21;
					task21->cl_arg = u21a;
			
					u21a->i = k+1;
					u21a->k = j;
					u21a->nblocks = args->nblocks;
					u21a->dataA = args->dataA;

					task21->buffers[0].handle = 
						starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->i);
					task21->buffers[0].mode = STARPU_R;
					task21->buffers[1].handle =
						starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->k);
					task21->buffers[1].mode = STARPU_RW;
		
					starpu_task_submit(task21);
				}
		}
	}

	/* 11k + 22k-1,i,k => 12 k,i */
	if (j == k + 1) {
		uint8_t dep;
		/* 11 k+1*/
		dep = advance_11[(k+1)];
		if (dep & DONE) {
			/* try to push the task */
			uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[(k+1)*nblocks + i], STARTED);
				 if ((u & STARTED) == 0) {
					/* we are the only one that should launch that task */
					cl_args *u12a = malloc(sizeof(cl_args));

					struct starpu_task *task12 = starpu_task_create();
						task12->callback_func = dw_callback_v2_codelet_update_u12;
						task12->callback_arg = u12a;
						task12->cl = &cl12;
						task12->cl_arg = u12a;

					u12a->i = k+1;
					u12a->k = i;
					u12a->nblocks = args->nblocks;
					u12a->dataA = args->dataA;

					task12->buffers[0].handle = starpu_data_get_sub_data(args->dataA, 2, u12a->i, u12a->i); 
					task12->buffers[0].mode = STARPU_R;
					task12->buffers[1].handle = starpu_data_get_sub_data(args->dataA, 2, u12a->k, u12a->i); 
					task12->buffers[1].mode = STARPU_RW;
					
					starpu_task_submit(task12);
				}
		}
	}

	free(args);
}

void dw_callback_v2_codelet_update_u12(void *argcb)
{
	cl_args *args = argcb;	

	/* now launch the update of LU22 */
	unsigned i = args->i;
	unsigned k = args->k;
	unsigned nblocks = args->nblocks;

	/* we did task 21i,k */
	advance_12_21[i*nblocks + k] = DONE;

	unsigned slicey;
	for (slicey = i+1; slicey < nblocks; slicey++)
	{
		/* can we launch 22 i,args->k,slicey ? */
		/* deps : 21 args->k, slicey */
		uint8_t dep;
		dep = advance_12_21[i + slicey*nblocks];
		if (dep & DONE)
		{
			/* perhaps we may schedule the 22 i,args->k,slicey task */
			uint8_t u = STARPU_ATOMIC_OR(&advance_22[i*nblocks*nblocks + slicey*nblocks + k], STARTED);
                        if ((u & STARTED) == 0) {
				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));

				struct starpu_task *task22 = starpu_task_create();
				task22->callback_func = dw_callback_v2_codelet_update_u22;
				task22->callback_arg = u22a;
				task22->cl = &cl22;
				task22->cl_arg = u22a;

				u22a->k = i;
				u22a->i = k;
				u22a->j = slicey;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;

				task22->buffers[0].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->k);
				task22->buffers[0].mode = STARPU_R;

				task22->buffers[1].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->k, u22a->j);
				task22->buffers[1].mode = STARPU_R;

				task22->buffers[2].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->j);
				task22->buffers[2].mode = STARPU_RW;
				
				/* schedule that codelet */
				if (!no_prio && (slicey == i+1))
					task22->priority = STARPU_MAX_PRIO;

				starpu_task_submit(task22);
			}
		}
	}
}

void dw_callback_v2_codelet_update_u21(void *argcb)
{
	cl_args *args = argcb;	

	/* now launch the update of LU22 */
	unsigned i = args->i;
	unsigned k = args->k;
	unsigned nblocks = args->nblocks;

	/* we did task 21i,k */
	advance_12_21[i + k*nblocks] = DONE;


	unsigned slicex;
	for (slicex = i+1; slicex < nblocks; slicex++)
	{
		/* can we launch 22 i,slicex,k ? */
		/* deps : 12 slicex k */
		uint8_t dep;
		dep = advance_12_21[i*nblocks + slicex];
		if (dep & DONE)
		{
			/* perhaps we may schedule the 22 i,args->k,slicey task */
			uint8_t u = STARPU_ATOMIC_OR(&advance_22[i*nblocks*nblocks + k*nblocks + slicex], STARTED);
                        if ((u & STARTED) == 0) {
				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));

				struct starpu_task *task22 = starpu_task_create();
				task22->callback_func = dw_callback_v2_codelet_update_u22;
				task22->callback_arg = u22a;
				task22->cl = &cl22;
				task22->cl_arg = u22a;

				u22a->k = i;
				u22a->i = slicex;
				u22a->j = k;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;

				task22->buffers[0].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->k);
				task22->buffers[0].mode = STARPU_R;

				task22->buffers[1].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->k, u22a->j);
				task22->buffers[1].mode = STARPU_R;

				task22->buffers[2].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->j);
				task22->buffers[2].mode = STARPU_RW;
				
				/* schedule that codelet */
				if (!no_prio && (slicex == i+1))
					task22->priority = STARPU_MAX_PRIO;

				starpu_task_submit(task22);
			}
		}
	}
}

void dw_callback_v2_codelet_update_u11(void *argcb)
{
	/* in case there remains work, go on */
	cl_args *args = argcb;

	unsigned nblocks = args->nblocks;
	unsigned i = args->i;

	/* we did task 11k */
	advance_11[i] = DONE;

	if (i == nblocks - 1) 
	{
		/* we are done : wake the application up  */
		pthread_mutex_lock(&mutex);
		finished = 1;
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
		return;
	}
	else 
	{
		/* put new tasks */
		unsigned slice;
		for (slice = i + 1; slice < nblocks; slice++)
		{

			/* can we launch 12i,slice ? */
			uint8_t deps12;
			if (i == 0) {
				deps12 = DONE;
			}
			else {
				deps12 = advance_22[(i-1)*nblocks*nblocks + slice + i*nblocks];		
			}
			if (deps12 & DONE) {
				/* we may perhaps launch the task 12i,slice */
				 uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[i*nblocks + slice], STARTED);
				 if ((u & STARTED) == 0) {
					/* we are the only one that should launch that task */
					cl_args *u12a = malloc(sizeof(cl_args));

					struct starpu_task *task12 = starpu_task_create();
						task12->callback_func = dw_callback_v2_codelet_update_u12;
						task12->callback_arg = u12a;
						task12->cl = &cl12;
						task12->cl_arg = u12a;

					u12a->i = i;
					u12a->k = slice;
					u12a->nblocks = args->nblocks;
					u12a->dataA = args->dataA;

					task12->buffers[0].handle = starpu_data_get_sub_data(args->dataA, 2, u12a->i, u12a->i); 
					task12->buffers[0].mode = STARPU_R;
					task12->buffers[1].handle = starpu_data_get_sub_data(args->dataA, 2, u12a->k, u12a->i); 
					task12->buffers[1].mode = STARPU_RW;

					if (!no_prio && (slice == i +1))
						task12->priority = STARPU_MAX_PRIO;

					starpu_task_submit(task12);
				}
			}

			/* can we launch 21i,slice ? */
			if (i == 0) {
				deps12 = DONE;
			}
			else {
				deps12 = advance_22[(i-1)*nblocks*nblocks + slice*nblocks + i];		
			}
			if (deps12 & DONE) {
				/* we may perhaps launch the task 12i,slice */
				 uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[i + slice*nblocks], STARTED);
				 if ((u & STARTED) == 0) {
					/* we are the only one that should launch that task */
					cl_args *u21a = malloc(sizeof(cl_args));

					struct starpu_task *task21 = starpu_task_create();
						task21->callback_func = dw_callback_v2_codelet_update_u21;
						task21->callback_arg = u21a;
						task21->cl = &cl21;
						task21->cl_arg = u21a;
		
					u21a->i = i;
					u21a->k = slice;
					u21a->nblocks = args->nblocks;
					u21a->dataA = args->dataA;

					task21->buffers[0].handle = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->i);
					task21->buffers[0].mode = STARPU_R;
					task21->buffers[1].handle = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->k);
					task21->buffers[1].mode = STARPU_RW;
		
					if (!no_prio && (slice == i +1))
						task21->priority = STARPU_MAX_PRIO;

					starpu_task_submit(task21);
				}
			}
		}
	}
}



/*
 *	Callbacks 
 */


void dw_callback_codelet_update_u11(void *argcb)
{
	/* in case there remains work, go on */
	cl_args *args = argcb;

	if (args->i == args->nblocks - 1) 
	{
		/* we are done : wake the application up  */
		pthread_mutex_lock(&mutex);
		finished = 1;
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
		return;
	}
	else 
	{
		/* put new tasks */
		unsigned nslices;
		nslices = args->nblocks - 1 - args->i;

		unsigned *remaining = malloc(sizeof(unsigned));
		*remaining = 2*nslices; 

		unsigned slice;
		for (slice = args->i + 1; slice < args->nblocks; slice++)
		{

			/* update slice from u12 */
			cl_args *u12a = malloc(sizeof(cl_args));

			/* update slice from u21 */
			cl_args *u21a = malloc(sizeof(cl_args));

			struct starpu_task *task12 = starpu_task_create();
				task12->callback_func = dw_callback_codelet_update_u12_21;
				task12->callback_arg = u12a;
				task12->cl = &cl12;
				task12->cl_arg = u12a;

			struct starpu_task *task21 = starpu_task_create();
				task21->callback_func = dw_callback_codelet_update_u12_21;
				task21->callback_arg = u21a;
				task21->cl = &cl21;
				task21->cl_arg = u21a;
			
			u12a->i = args->i;
			u12a->k = slice;
			u12a->nblocks = args->nblocks;
			u12a->dataA = args->dataA;
			u12a->remaining = remaining;
			
			u21a->i = args->i;
			u21a->k = slice;
			u21a->nblocks = args->nblocks;
			u21a->dataA = args->dataA;
			u21a->remaining = remaining;

			task12->buffers[0].handle = 
				starpu_data_get_sub_data(args->dataA, 2, u12a->i, u12a->i); 
			task12->buffers[0].mode = STARPU_R;
			task12->buffers[1].handle = 
				starpu_data_get_sub_data(args->dataA, 2, u12a->k, u12a->i); 
			task12->buffers[1].mode = STARPU_RW;

			task21->buffers[0].handle = 
				starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->i);
			task21->buffers[0].mode = STARPU_R;
			task21->buffers[1].handle = 
				starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->k);
			task21->buffers[1].mode = STARPU_RW;
		
			starpu_task_submit(task12);
			starpu_task_submit(task21);
		}
	}
}


void dw_callback_codelet_update_u22(void *argcb)
{
	cl_args *args = argcb;	

	if (STARPU_ATOMIC_ADD(args->remaining, (-1)) == 0)
	{
		/* all worker already used the counter */
		free(args->remaining);

		/* we now reduce the LU22 part (recursion appears there) */
		cl_args *u11arg = malloc(sizeof(cl_args));
	
		struct starpu_task *task = starpu_task_create();
			task->callback_func = dw_callback_codelet_update_u11;
			task->callback_arg = u11arg;
			task->cl = &cl11;
			task->cl_arg = u11arg;

			task->buffers[0].handle = starpu_data_get_sub_data(args->dataA, 2, args->k + 1, args->k + 1);
			task->buffers[0].mode = STARPU_RW;
	
		u11arg->dataA = args->dataA;
		u11arg->i = args->k + 1;
		u11arg->nblocks = args->nblocks;

		/* schedule the codelet */
		starpu_task_submit(task);
	}

	free(args);
}

void dw_callback_codelet_update_u12_21(void *argcb)
{
	cl_args *args = argcb;	

	if (STARPU_ATOMIC_ADD(args->remaining, -1) == 0)
	{
		/* now launch the update of LU22 */
		unsigned i = args->i;
		unsigned nblocks = args->nblocks;

		/* the number of tasks to be done */
		unsigned *remaining = malloc(sizeof(unsigned));
		*remaining = (nblocks - 1 - i)*(nblocks - 1 - i);

		unsigned slicey, slicex;
		for (slicey = i+1; slicey < nblocks; slicey++)
		{
			for (slicex = i+1; slicex < nblocks; slicex++)
			{
				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));

				struct starpu_task *task22 = starpu_task_create();
				task22->callback_func = dw_callback_codelet_update_u22;
				task22->callback_arg = u22a;
				task22->cl = &cl22;
				task22->cl_arg = u22a;

				u22a->k = i;
				u22a->i = slicex;
				u22a->j = slicey;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;
				u22a->remaining = remaining;

				task22->buffers[0].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->k);
				task22->buffers[0].mode = STARPU_R;

				task22->buffers[1].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->k, u22a->j);
				task22->buffers[1].mode = STARPU_R;

				task22->buffers[2].handle = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->j);
				task22->buffers[2].mode = STARPU_RW;
				
				/* schedule that codelet */
				starpu_task_submit(task22);
			}
		}
	}
}



/*
 *	code to bootstrap the factorization 
 */

void dw_codelet_facto(starpu_data_handle dataA, unsigned nblocks)
{
	cl_args *args = malloc(sizeof(cl_args));

	args->i = 0;
	args->nblocks = nblocks;
	args->dataA = dataA;

	gettimeofday(&start, NULL);

	/* inject a new task with this codelet into the system */ 
	struct starpu_task *task = starpu_task_create();
		task->callback_func = dw_callback_codelet_update_u11;
		task->callback_arg = args;
		task->cl = &cl11;
		task->cl_arg = args;

		task->buffers[0].handle = starpu_data_get_sub_data(dataA, 2, 0, 0);
		task->buffers[0].mode = STARPU_RW;

	/* schedule the codelet */
	starpu_task_submit(task);

	/* stall the application until the end of computations */
	pthread_mutex_lock(&mutex);

	if (!finished)
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;
	fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

void dw_codelet_facto_v2(starpu_data_handle dataA, unsigned nblocks)
{

	advance_11 = calloc(nblocks, sizeof(uint8_t));
	STARPU_ASSERT(advance_11);

	advance_12_21 = calloc(nblocks*nblocks, sizeof(uint8_t));
	STARPU_ASSERT(advance_12_21);

	advance_22 = calloc(nblocks*nblocks*nblocks, sizeof(uint8_t));
	STARPU_ASSERT(advance_22);

	cl_args *args = malloc(sizeof(cl_args));

	args->i = 0;
	args->nblocks = nblocks;
	args->dataA = dataA;

	gettimeofday(&start, NULL);

	/* inject a new task with this codelet into the system */ 
	struct starpu_task *task = starpu_task_create();
		task->callback_func = dw_callback_v2_codelet_update_u11;
		task->callback_arg = args;
		task->cl = &cl11;
		task->cl_arg = args;

		task->buffers[0].handle = starpu_data_get_sub_data(dataA, 2, 0, 0); 
		task->buffers[0].mode = STARPU_RW;

	/* schedule the codelet */
	int ret = starpu_task_submit(task);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		fprintf(stderr, "No worker may execute this task\n");
		exit(0);
	}

	/* stall the application until the end of computations */
	pthread_mutex_lock(&mutex);

	if (!finished)
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;
	fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

void initialize_system(float **A, float **B, unsigned dim, unsigned pinned)
{
	starpu_init(NULL);

	starpu_helper_cublas_init();

	if (pinned)
	{
		starpu_data_malloc_pinned_if_possible((void **)A, (size_t)dim*dim*sizeof(float));
		starpu_data_malloc_pinned_if_possible((void **)B, (size_t)dim*sizeof(float));
	} 
	else {
		*A = malloc((size_t)dim*dim*sizeof(float));
		STARPU_ASSERT(*A);
		*B = malloc((size_t)dim*sizeof(float));
		STARPU_ASSERT(*B);
	}
}

void dw_factoLU(float *matA, unsigned size, 
		unsigned ld, unsigned nblocks, 
		unsigned version, unsigned _no_prio)
{

#ifdef CHECK_RESULTS
	fprintf(stderr, "Checking results ...\n");
	float *Asaved;
	Asaved = malloc((size_t)ld*ld*sizeof(float));

	memcpy(Asaved, matA, (size_t)ld*ld*sizeof(float));
#endif

	no_prio = _no_prio;

	starpu_data_handle dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, 0, (uintptr_t)matA, ld, 
			size, size, sizeof(float));

	struct starpu_data_filter f;
		f.filter_func = starpu_vertical_block_filter_func;
		f.nchildren = nblocks;
		f.get_nchildren = NULL;
		f.get_child_ops = NULL;

	struct starpu_data_filter f2;
		f2.filter_func = starpu_block_filter_func;
		f2.nchildren = nblocks;
		f2.get_nchildren = NULL;
		f2.get_child_ops = NULL;

	starpu_data_map_filters(dataA, 2, &f, &f2);

	switch (version) {
		case 1:
			dw_codelet_facto(dataA, nblocks);
			break;
		default:
		case 2:
			dw_codelet_facto_v2(dataA, nblocks);
			break;
	}

	/* gather all the data */
	starpu_data_unpartition(dataA, 0);

	starpu_data_unregister(dataA);

#ifdef CHECK_RESULTS
	compare_A_LU(Asaved, matA, size, ld);
#endif
}
