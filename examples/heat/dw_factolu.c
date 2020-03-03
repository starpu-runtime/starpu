/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2013       Thibaut Lambert
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
 * This implements an LU factorization.
 * The task graph is submitted through continuation: the rest of the graph is
 * submitted as appropriate in the tasks' callback.
 */

#include "dw_factolu.h"

#ifdef STARPU_HAVE_HELGRIND_H
#include <valgrind/helgrind.h>
#endif
#ifndef ANNOTATE_HAPPENS_BEFORE
#define ANNOTATE_HAPPENS_BEFORE(obj) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_AFTER
#define ANNOTATE_HAPPENS_AFTER(obj) ((void)0)
#endif

#if 0
#define debug(fmt, ...) fprintf(stderr, fmt, ## __VA_ARGS__)
#else
#define debug(fmt, ...)
#endif

struct starpu_perfmodel model_11;
struct starpu_perfmodel model_12;
struct starpu_perfmodel model_21;
struct starpu_perfmodel model_22;

static unsigned *advance_11; /* size nblocks, whether the 11 task is done */
static unsigned *advance_12_21; /* size nblocks*nblocks */
static unsigned *advance_22; /* array of nblocks *nblocks*nblocks */

static double start;
static double end;

static unsigned no_prio = 0;

static struct starpu_codelet cl11 =
{
	.cpu_funcs = {dw_cpu_codelet_update_u11},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u11"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u11},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &model_11
};

static struct starpu_codelet cl12 =
{
	.cpu_funcs = {dw_cpu_codelet_update_u12},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u12"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u12},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &model_12
};

static struct starpu_codelet cl21 =
{
	.cpu_funcs = {dw_cpu_codelet_update_u21},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u21"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u21},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &model_21
};

static struct starpu_codelet cl22 =
{
	.cpu_funcs = {dw_cpu_codelet_update_u22},
	.cpu_funcs_name = {"dw_cpu_codelet_update_u22"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_u22},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &model_22
};



#define STARTED	0x01
#define DONE	0x11

/*
 *	Upgraded Callbacks : break the pipeline design !
 */

void dw_callback_v2_codelet_update_u22(void *argcb)
{
	int ret;
	cl_args *args = argcb;

	unsigned k = args->k;
	unsigned i = args->i;
	unsigned j = args->j;
	unsigned nblocks = args->nblocks;

	debug("u22 %d %d %d\n", k, i, j);

	/* we did task 22k,i,j */
	advance_22[k*nblocks*nblocks + i + j*nblocks] = DONE;

	if ( (i == j) && (i == k+1))
	{
		/* we now reduce the LU22 part (recursion appears there) */
		cl_args *u11arg = malloc(sizeof(cl_args));

		struct starpu_task *task = starpu_task_create();
		task->callback_func = dw_callback_v2_codelet_update_u11;
		task->callback_arg = u11arg;
		task->cl = &cl11;
		task->cl_arg = u11arg;
		task->cl_arg_size = sizeof(*u11arg);

		task->handles[0] = starpu_data_get_sub_data(args->dataA, 2, k+1, k+1);

		u11arg->dataA = args->dataA;
		u11arg->i = k + 1;
		u11arg->nblocks = args->nblocks;

		/* schedule the codelet */
		if (!no_prio)
			task->priority = STARPU_MAX_PRIO;

		debug( "u22 %d %d %d start u11 %d\n", k, i, j, k + 1);
		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* 11k+1 + 22k,k+1,j => 21 k+1,j */
	if ( i == k + 1 && j > k + 1)
	{
		uint8_t dep;
		/* 11 k+1*/
		dep = advance_11[(k+1)];
		if (dep & DONE)
		{
			/* try to push the task */
			uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[(k+1) + j*nblocks], STARTED);
				if ((u & STARTED) == 0)
				{
					/* we are the only one that should launch that task */
					cl_args *u21a = malloc(sizeof(cl_args));

					struct starpu_task *task21 = starpu_task_create();
					task21->callback_func = dw_callback_v2_codelet_update_u21;
					task21->callback_arg = u21a;
					task21->cl = &cl21;
					task21->cl_arg = u21a;
					task21->cl_arg_size = sizeof(*u21a);

					u21a->i = k+1;
					u21a->k = j;
					u21a->nblocks = args->nblocks;
					u21a->dataA = args->dataA;

					task21->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->i);
					task21->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->k);

					debug( "u22 %d %d %d start u21 %d %d\n", k, i, j, k+1, j);
					ret = starpu_task_submit(task21);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
				}
		}
	}

	/* 11k + 22k-1,i,k => 12 k,i */
	if (j == k + 1 && i > k + 1)
	{
		uint8_t dep;
		/* 11 k+1*/
		dep = advance_11[(k+1)];
		if (dep & DONE)
		{
			/* try to push the task */
			uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[(k+1)*nblocks + i], STARTED);
				 if ((u & STARTED) == 0)
				 {
					/* we are the only one that should launch that task */
					cl_args *u12a = malloc(sizeof(cl_args));

					struct starpu_task *task12 = starpu_task_create();
						task12->callback_func = dw_callback_v2_codelet_update_u12;
						task12->callback_arg = u12a;
						task12->cl = &cl12;
						task12->cl_arg = u12a;
						task12->cl_arg_size = sizeof(*u12a);

					u12a->i = k+1;
					u12a->k = i;
					u12a->nblocks = args->nblocks;
					u12a->dataA = args->dataA;

					task12->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u12a->i, u12a->i);
					task12->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u12a->k, u12a->i);

					debug( "u22 %d %d %d start u12 %d %d\n", k, i, j, k+1, i);
					ret = starpu_task_submit(task12);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
				}
		}
	}

	free(args);
}

void dw_callback_v2_codelet_update_u12(void *argcb)
{
	int ret;
	cl_args *args = argcb;

	/* now launch the update of LU22 */
	unsigned i = args->i;
	unsigned k = args->k;
	unsigned nblocks = args->nblocks;

	debug( "u12 %d %d\n", i, k);

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
                        if ((u & STARTED) == 0)
			{
				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));

				struct starpu_task *task22 = starpu_task_create();
				task22->callback_func = dw_callback_v2_codelet_update_u22;
				task22->callback_arg = u22a;
				task22->cl = &cl22;
				task22->cl_arg = u22a;
				task22->cl_arg_size = sizeof(*u22a);

				u22a->k = i;
				u22a->i = k;
				u22a->j = slicey;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;

				task22->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->k);
				task22->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u22a->k, u22a->j);
				task22->handles[2] = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->j);

				/* schedule that codelet */
				if (!no_prio && (slicey == i+1))
					task22->priority = STARPU_MAX_PRIO;

				debug( "u12 %d %d start u22 %d %d %d\n", i, k, i, k, slicey);
				ret = starpu_task_submit(task22);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			}
		}
	}
	free(argcb);
}

void dw_callback_v2_codelet_update_u21(void *argcb)
{
	int ret;
	cl_args *args = argcb;

	/* now launch the update of LU22 */
	unsigned i = args->i;
	unsigned k = args->k;
	unsigned nblocks = args->nblocks;

	/* we did task 21i,k */
	advance_12_21[i + k*nblocks] = DONE;

	debug("u21 %d %d\n", i, k);

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
                        if ((u & STARTED) == 0)
			{
				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));

				struct starpu_task *task22 = starpu_task_create();
				task22->callback_func = dw_callback_v2_codelet_update_u22;
				task22->callback_arg = u22a;
				task22->cl = &cl22;
				task22->cl_arg = u22a;
				task22->cl_arg_size = sizeof(*u22a);

				u22a->k = i;
				u22a->i = slicex;
				u22a->j = k;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;

				task22->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->k);
				task22->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u22a->k, u22a->j);
				task22->handles[2] = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->j);

				/* schedule that codelet */
				if (!no_prio && (slicex == i+1))
					task22->priority = STARPU_MAX_PRIO;

				debug( "u21 %d %d start u22 %d %d %d\n", i, k, i, slicex, k);
				ret = starpu_task_submit(task22);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			}
		}
	}
	free(argcb);
}

void dw_callback_v2_codelet_update_u11(void *argcb)
{
	/* in case there remains work, go on */
	cl_args *args = argcb;

	unsigned nblocks = args->nblocks;
	unsigned i = args->i;

	debug("u11 %d\n", i);

	/* we did task 11k */
	advance_11[i] = DONE;

	if (i == nblocks - 1)
	{
		/* we are done */
		free(argcb);
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
			if (i == 0)
			{
				deps12 = DONE;
			}
			else
			{
				deps12 = advance_22[(i-1)*nblocks*nblocks + slice + i*nblocks];
			}
			if (deps12 & DONE)
			{
				/* we may perhaps launch the task 12i,slice */
				uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[i*nblocks + slice], STARTED);
				if ((u & STARTED) == 0)
				{
					int ret;

					/* we are the only one that should launch that task */
					cl_args *u12a = malloc(sizeof(cl_args));

					struct starpu_task *task12 = starpu_task_create();
					task12->callback_func = dw_callback_v2_codelet_update_u12;
					task12->callback_arg = u12a;
					task12->cl = &cl12;
					task12->cl_arg = u12a;
					task12->cl_arg_size = sizeof(*u12a);

					u12a->i = i;
					u12a->k = slice;
					u12a->nblocks = args->nblocks;
					u12a->dataA = args->dataA;

					task12->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u12a->i, u12a->i);
					task12->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u12a->k, u12a->i);

					if (!no_prio && (slice == i +1))
						task12->priority = STARPU_MAX_PRIO;

					debug( "u11 %d start u12 %d %d\n", i, i, slice);
					ret = starpu_task_submit(task12);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
				}
			}

			/* can we launch 21i,slice ? */
			if (i == 0)
			{
				deps12 = DONE;
			}
			else
			{
				deps12 = advance_22[(i-1)*nblocks*nblocks + slice*nblocks + i];
			}
			if (deps12 & DONE)
			{
				/* we may perhaps launch the task 12i,slice */
				uint8_t u = STARPU_ATOMIC_OR(&advance_12_21[i + slice*nblocks], STARTED);
				if ((u & STARTED) == 0)
				{
					int ret;

					/* we are the only one that should launch that task */
					cl_args *u21a = malloc(sizeof(cl_args));

					struct starpu_task *task21 = starpu_task_create();
					task21->callback_func = dw_callback_v2_codelet_update_u21;
					task21->callback_arg = u21a;
					task21->cl = &cl21;
					task21->cl_arg = u21a;
					task21->cl_arg_size = sizeof(*u21a);

					u21a->i = i;
					u21a->k = slice;
					u21a->nblocks = args->nblocks;
					u21a->dataA = args->dataA;

					task21->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->i);
					task21->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->k);

					if (!no_prio && (slice == i +1))
						task21->priority = STARPU_MAX_PRIO;

					debug( "u11 %d start u21 %d %d\n", i, i, slice);
					ret = starpu_task_submit(task21);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
				}
			}
		}
	}
	free(argcb);
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
		/* we are done */
		free(argcb);
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
			int ret;

			/* update slice from u12 */
			cl_args *u12a = malloc(sizeof(cl_args));

			/* update slice from u21 */
			cl_args *u21a = malloc(sizeof(cl_args));

			struct starpu_task *task12 = starpu_task_create();
			task12->callback_func = dw_callback_codelet_update_u12_21;
			task12->callback_arg = u12a;
			task12->cl = &cl12;
			task12->cl_arg = u12a;
			task12->cl_arg_size = sizeof(*u12a);

			struct starpu_task *task21 = starpu_task_create();
			task21->callback_func = dw_callback_codelet_update_u12_21;
			task21->callback_arg = u21a;
			task21->cl = &cl21;
			task21->cl_arg = u21a;
			task21->cl_arg_size = sizeof(*u21a);

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

			task12->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u12a->i, u12a->i);
			task12->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u12a->k, u12a->i);

			task21->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->i);
			task21->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u21a->i, u21a->k);

			ret = starpu_task_submit(task12);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			ret = starpu_task_submit(task21);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		free(remaining);
	}
}


void dw_callback_codelet_update_u22(void *argcb)
{
	cl_args *args = argcb;
	unsigned remaining = STARPU_ATOMIC_ADD(args->remaining, (-1));
	ANNOTATE_HAPPENS_BEFORE(args->remaining);

	if (remaining == 0)
	{
		int ret;

		ANNOTATE_HAPPENS_AFTER(args->remaining);
		/* all worker already used the counter */
		free(args->remaining);

		/* we now reduce the LU22 part (recursion appears there) */
		cl_args *u11arg = malloc(sizeof(cl_args));

		struct starpu_task *task = starpu_task_create();
		task->callback_func = dw_callback_codelet_update_u11;
		task->callback_arg = u11arg;
		task->cl = &cl11;
		task->cl_arg = u11arg;
		task->cl_arg_size = sizeof(*u11arg);

		task->handles[0] = starpu_data_get_sub_data(args->dataA, 2, args->k + 1, args->k + 1);

		u11arg->dataA = args->dataA;
		u11arg->i = args->k + 1;
		u11arg->nblocks = args->nblocks;

		/* schedule the codelet */
		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	free(args);
}

void dw_callback_codelet_update_u12_21(void *argcb)
{
	cl_args *args = argcb;
	unsigned remaining = STARPU_ATOMIC_ADD(args->remaining, -1);
	ANNOTATE_HAPPENS_BEFORE(args->remaining);

	if (remaining == 0)
	{
		ANNOTATE_HAPPENS_AFTER(args->remaining);
		/* now launch the update of LU22 */
		unsigned i = args->i;
		unsigned nblocks = args->nblocks;

		/* the number of tasks to be done */
		unsigned *remaining_tasks = malloc(sizeof(unsigned));
		*remaining_tasks = (nblocks - 1 - i)*(nblocks - 1 - i);

		unsigned slicey, slicex;
		for (slicey = i+1; slicey < nblocks; slicey++)
		{
			for (slicex = i+1; slicex < nblocks; slicex++)
			{
				int ret;

				/* update that square matrix */
				cl_args *u22a = malloc(sizeof(cl_args));

				struct starpu_task *task22 = starpu_task_create();
				task22->callback_func = dw_callback_codelet_update_u22;
				task22->callback_arg = u22a;
				task22->cl = &cl22;
				task22->cl_arg = u22a;
				task22->cl_arg_size = sizeof(*u22a);

				u22a->k = i;
				u22a->i = slicex;
				u22a->j = slicey;
				u22a->dataA = args->dataA;
				u22a->nblocks = nblocks;
				u22a->remaining = remaining_tasks;

				task22->handles[0] = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->k);
				task22->handles[1] = starpu_data_get_sub_data(args->dataA, 2, u22a->k, u22a->j);
				task22->handles[2] = starpu_data_get_sub_data(args->dataA, 2, u22a->i, u22a->j);

				/* schedule that codelet */
				ret = starpu_task_submit(task22);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			}
		}

		free(remaining_tasks);
	}
}



/*
 *	code to bootstrap the factorization
 */

void dw_codelet_facto(starpu_data_handle_t dataA, unsigned nblocks)
{
	int ret;
	cl_args *args = malloc(sizeof(cl_args));

	args->i = 0;
	args->nblocks = nblocks;
	args->dataA = dataA;

	start = starpu_timing_now();

	/* inject a new task with this codelet into the system */
	struct starpu_task *task = starpu_task_create();
	task->callback_func = dw_callback_codelet_update_u11;
	task->callback_arg = args;
	task->cl = &cl11;
	task->cl_arg = args;

	task->handles[0] = starpu_data_get_sub_data(dataA, 2, 0, 0);

	/* schedule the codelet */
	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();

	end = starpu_timing_now();

	double timing = end - start;

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;

	PRINTF("# size\tms\tGFlops\n");
	PRINTF("%u\t%.0f\t%.1f\n", n, timing/1000, flop/timing/1000.0f);
}

void dw_codelet_facto_v2(starpu_data_handle_t dataA, unsigned nblocks)
{

	advance_11 = calloc(nblocks, sizeof(*advance_11));
	STARPU_ASSERT(advance_11);

	advance_12_21 = calloc(nblocks*nblocks, sizeof(*advance_12_21));
	STARPU_ASSERT(advance_12_21);

	advance_22 = calloc(nblocks*nblocks*nblocks, sizeof(*advance_22));
	STARPU_ASSERT(advance_22);

	cl_args *args = malloc(sizeof(cl_args));

	args->i = 0;
	args->nblocks = nblocks;
	args->dataA = dataA;

	start = starpu_timing_now();

	/* inject a new task with this codelet into the system */
	struct starpu_task *task = starpu_task_create();
	task->callback_func = dw_callback_v2_codelet_update_u11;
	task->callback_arg = args;
	task->cl = &cl11;
	task->cl_arg = args;
	task->cl_arg_size = sizeof(*args);

	task->handles[0] = starpu_data_get_sub_data(dataA, 2, 0, 0);

	/* schedule the codelet */
	int ret = starpu_task_submit(task);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		FPRINTF(stderr, "No worker may execute this task\n");
		exit(0);
	}

	starpu_task_wait_for_all();

	end = starpu_timing_now();

	double timing = end - start;

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;

	PRINTF("# size\tms\tGFlops\n");
	PRINTF("%u\t%.0f\t%.1f\n", n, timing/1000, flop/timing/1000.0f);

	free(advance_11);
	free(advance_12_21);
	free(advance_22);
}

void initialize_system(float **A, float **B, unsigned dim, unsigned pinned)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_ATLAS
	char * symbol_11 = "lu_model_11_atlas";
	char * symbol_12 = "lu_model_12_atlas";
	char * symbol_21 = "lu_model_21_atlas";
	char * symbol_22 = "lu_model_22_atlas";
#elif defined(STARPU_GOTO)
	char * symbol_11 = "lu_model_11_goto";
	char * symbol_12 = "lu_model_12_goto";
	char * symbol_21 = "lu_model_21_goto";
	char * symbol_22 = "lu_model_22_goto";
#elif defined(STARPU_OPENBLAS)
	char * symbol_11 = "lu_model_11_openblas";
	char * symbol_12 = "lu_model_12_openblas";
	char * symbol_21 = "lu_model_21_openblas";
	char * symbol_22 = "lu_model_22_openblas";
#else
	char * symbol_11 = "lu_model_11";
	char * symbol_12 = "lu_model_12";
	char * symbol_21 = "lu_model_21";
	char * symbol_22 = "lu_model_22";
#endif
	initialize_lu_kernels_model(&model_11,symbol_11,task_11_cost,task_11_cost_cpu,task_11_cost_cuda);
	initialize_lu_kernels_model(&model_12,symbol_12,task_12_cost,task_12_cost_cpu,task_12_cost_cuda);
	initialize_lu_kernels_model(&model_21,symbol_21,task_21_cost,task_21_cost_cpu,task_21_cost_cuda);
	initialize_lu_kernels_model(&model_22,symbol_22,task_22_cost,task_22_cost_cpu,task_22_cost_cuda);

	starpu_cublas_init();

	if (pinned)
	{
		starpu_malloc((void **)A, (size_t)dim*dim*sizeof(float));
		starpu_malloc((void **)B, (size_t)dim*sizeof(float));
	}
	else
	{
		*A = malloc((size_t)dim*dim*sizeof(float));
		STARPU_ASSERT(*A);
		*B = malloc((size_t)dim*sizeof(float));
		STARPU_ASSERT(*B);
	}
}

void free_system(float *A, float *B, unsigned pinned)
{
	if (pinned)
	{
		starpu_free(A);
		starpu_free(B);
	}
	else
	{
		free(A);
		free(B);
	}
}

void dw_factoLU(float *matA, unsigned size,
		unsigned ld, unsigned nblocks,
		unsigned version, unsigned _no_prio)
{

#ifdef CHECK_RESULTS
	FPRINTF(stderr, "Checking results ...\n");
	float *Asaved;
	Asaved = malloc((size_t)ld*ld*sizeof(float));

	memcpy(Asaved, matA, (size_t)ld*ld*sizeof(float));
#endif

	no_prio = _no_prio;

	starpu_data_handle_t dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, ld,
			size, size, sizeof(float));

	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nblocks
	};

	struct starpu_data_filter f2 =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = nblocks
	};

	starpu_data_map_filters(dataA, 2, &f, &f2);

	switch (version)
	{
		case 1:
			dw_codelet_facto(dataA, nblocks);
			break;
		default:
		case 2:
			dw_codelet_facto_v2(dataA, nblocks);
			break;
	}

	/* gather all the data */
	starpu_data_unpartition(dataA, STARPU_MAIN_RAM);

	starpu_data_unregister(dataA);

#ifdef CHECK_RESULTS
	compare_A_LU(Asaved, matA, size, ld);
#endif
}
