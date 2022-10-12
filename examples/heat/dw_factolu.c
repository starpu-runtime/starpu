/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

struct starpu_perfmodel model_getrf;
struct starpu_perfmodel model_trsm_ll;
struct starpu_perfmodel model_trsm_ru;
struct starpu_perfmodel model_gemm;

static unsigned *advance_11; /* size nblocks, whether the 11 task is done */
static unsigned *advance_12_21; /* size nblocks*nblocks */
static unsigned *advance_22; /* array of nblocks *nblocks*nblocks */

static double start;
static double end;

static unsigned no_prio = 0;

static struct starpu_codelet cl_getrf =
{
	.cpu_funcs = {dw_cpu_codelet_update_getrf},
	.cpu_funcs_name = {"dw_cpu_codelet_update_getrf"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_getrf},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &model_getrf
};

static struct starpu_codelet cl_trsm_ll =
{
	.cpu_funcs = {dw_cpu_codelet_update_trsm_ll},
	.cpu_funcs_name = {"dw_cpu_codelet_update_trsm_ll"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_trsm_ll},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &model_trsm_ll
};

static struct starpu_codelet cl_trsm_ru =
{
	.cpu_funcs = {dw_cpu_codelet_update_trsm_ru},
	.cpu_funcs_name = {"dw_cpu_codelet_update_trsm_ru"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_trsm_ru},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &model_trsm_ru
};

static struct starpu_codelet cl_gemm =
{
	.cpu_funcs = {dw_cpu_codelet_update_gemm},
	.cpu_funcs_name = {"dw_cpu_codelet_update_gemm"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dw_cublas_codelet_update_gemm},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &model_gemm
};



#define STARTED	0x01
#define DONE	0x11

/*
 *	Upgraded Callbacks : break the pipeline design !
 */

void dw_callback_v2_codelet_update_gemm(void *argcb)
{
	int ret;
	cl_args *args = argcb;

	unsigned k = args->k;
	unsigned i = args->i;
	unsigned j = args->j;
	unsigned nblocks = args->nblocks;

	debug("ugemm %d %d %d\n", k, i, j);

	/* we did task 22k,i,j */
	advance_22[k*nblocks*nblocks + i + j*nblocks] = DONE;

	if ((i == j) && (i == k+1))
	{
		/* we now reduce the LU22 part (recursion appears there) */
		cl_args *ugetrfarg = malloc(sizeof(cl_args));

		struct starpu_task *task = starpu_task_create();
		task->callback_func = dw_callback_v2_codelet_update_getrf;
		task->callback_arg = ugetrfarg;
		task->cl = &cl_getrf;
		task->cl_arg = ugetrfarg;
		task->cl_arg_size = sizeof(*ugetrfarg);

		task->handles[0] = starpu_data_get_sub_data(args->dataA, 2, k+1, k+1);

		ugetrfarg->dataA = args->dataA;
		ugetrfarg->i = k + 1;
		ugetrfarg->nblocks = args->nblocks;

		/* schedule the codelet */
		if (!no_prio)
			task->priority = STARPU_MAX_PRIO;

		debug("ugemm %d %d %d start ugetrf %d\n", k, i, j, k + 1);
		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* 11k+1 + 22k,k+1,j => 21 k+1,j */
	if (i == k + 1 && j > k + 1)
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
					cl_args *utrsmrua = malloc(sizeof(cl_args));

					struct starpu_task *task_trsm_ru = starpu_task_create();
					task_trsm_ru->callback_func = dw_callback_v2_codelet_update_trsm_ru;
					task_trsm_ru->callback_arg = utrsmrua;
					task_trsm_ru->cl = &cl_trsm_ru;
					task_trsm_ru->cl_arg = utrsmrua;
					task_trsm_ru->cl_arg_size = sizeof(*utrsmrua);

					utrsmrua->i = k+1;
					utrsmrua->k = j;
					utrsmrua->nblocks = args->nblocks;
					utrsmrua->dataA = args->dataA;

					task_trsm_ru->handles[0] = starpu_data_get_sub_data(args->dataA, 2, utrsmrua->i, utrsmrua->i);
					task_trsm_ru->handles[1] = starpu_data_get_sub_data(args->dataA, 2, utrsmrua->i, utrsmrua->k);

					debug("ugemm %d %d %d start utrsmru %d %d\n", k, i, j, k+1, j);
					ret = starpu_task_submit(task_trsm_ru);
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
					cl_args *utrsmlla = malloc(sizeof(cl_args));

					struct starpu_task *task_trsm_ll = starpu_task_create();
						task_trsm_ll->callback_func = dw_callback_v2_codelet_update_trsm_ll;
						task_trsm_ll->callback_arg = utrsmlla;
						task_trsm_ll->cl = &cl_trsm_ll;
						task_trsm_ll->cl_arg = utrsmlla;
						task_trsm_ll->cl_arg_size = sizeof(*utrsmlla);

					utrsmlla->i = k+1;
					utrsmlla->k = i;
					utrsmlla->nblocks = args->nblocks;
					utrsmlla->dataA = args->dataA;

					task_trsm_ll->handles[0] = starpu_data_get_sub_data(args->dataA, 2, utrsmlla->i, utrsmlla->i);
					task_trsm_ll->handles[1] = starpu_data_get_sub_data(args->dataA, 2, utrsmlla->k, utrsmlla->i);

					debug("ugemm %d %d %d start utrsmll %d %d\n", k, i, j, k+1, i);
					ret = starpu_task_submit(task_trsm_ll);
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
				}
		}
	}

	free(args);
}

void dw_callback_v2_codelet_update_trsm_ll(void *argcb)
{
	int ret;
	cl_args *args = argcb;

	/* now launch the update of LU22 */
	unsigned i = args->i;
	unsigned k = args->k;
	unsigned nblocks = args->nblocks;

	debug("utrsmll %d %d\n", i, k);

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
				cl_args *ugemma = malloc(sizeof(cl_args));

				struct starpu_task *task_gemm = starpu_task_create();
				task_gemm->callback_func = dw_callback_v2_codelet_update_gemm;
				task_gemm->callback_arg = ugemma;
				task_gemm->cl = &cl_gemm;
				task_gemm->cl_arg = ugemma;
				task_gemm->cl_arg_size = sizeof(*ugemma);

				ugemma->k = i;
				ugemma->i = k;
				ugemma->j = slicey;
				ugemma->dataA = args->dataA;
				ugemma->nblocks = nblocks;

				task_gemm->handles[0] = starpu_data_get_sub_data(args->dataA, 2, ugemma->i, ugemma->k);
				task_gemm->handles[1] = starpu_data_get_sub_data(args->dataA, 2, ugemma->k, ugemma->j);
				task_gemm->handles[2] = starpu_data_get_sub_data(args->dataA, 2, ugemma->i, ugemma->j);

				/* schedule that codelet */
				if (!no_prio && (slicey == i+1))
					task_gemm->priority = STARPU_MAX_PRIO;

				debug("utrsmll %d %d start ugemm %d %d %d\n", i, k, i, k, slicey);
				ret = starpu_task_submit(task_gemm);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			}
		}
	}
	free(argcb);
}

void dw_callback_v2_codelet_update_trsm_ru(void *argcb)
{
	int ret;
	cl_args *args = argcb;

	/* now launch the update of LU22 */
	unsigned i = args->i;
	unsigned k = args->k;
	unsigned nblocks = args->nblocks;

	/* we did task 21i,k */
	advance_12_21[i + k*nblocks] = DONE;

	debug("utrsmru %d %d\n", i, k);

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
				cl_args *ugemma = malloc(sizeof(cl_args));

				struct starpu_task *task_gemm = starpu_task_create();
				task_gemm->callback_func = dw_callback_v2_codelet_update_gemm;
				task_gemm->callback_arg = ugemma;
				task_gemm->cl = &cl_gemm;
				task_gemm->cl_arg = ugemma;
				task_gemm->cl_arg_size = sizeof(*ugemma);

				ugemma->k = i;
				ugemma->i = slicex;
				ugemma->j = k;
				ugemma->dataA = args->dataA;
				ugemma->nblocks = nblocks;

				task_gemm->handles[0] = starpu_data_get_sub_data(args->dataA, 2, ugemma->i, ugemma->k);
				task_gemm->handles[1] = starpu_data_get_sub_data(args->dataA, 2, ugemma->k, ugemma->j);
				task_gemm->handles[2] = starpu_data_get_sub_data(args->dataA, 2, ugemma->i, ugemma->j);

				/* schedule that codelet */
				if (!no_prio && (slicex == i+1))
					task_gemm->priority = STARPU_MAX_PRIO;

				debug("utrsmru %d %d start ugemm %d %d %d\n", i, k, i, slicex, k);
				ret = starpu_task_submit(task_gemm);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			}
		}
	}
	free(argcb);
}

void dw_callback_v2_codelet_update_getrf(void *argcb)
{
	/* in case there remains work, go on */
	cl_args *args = argcb;

	unsigned nblocks = args->nblocks;
	unsigned i = args->i;

	debug("ugetrf %d\n", i);

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
					cl_args *utrsmlla = malloc(sizeof(cl_args));

					struct starpu_task *task_trsm_ll = starpu_task_create();
					task_trsm_ll->callback_func = dw_callback_v2_codelet_update_trsm_ll;
					task_trsm_ll->callback_arg = utrsmlla;
					task_trsm_ll->cl = &cl_trsm_ll;
					task_trsm_ll->cl_arg = utrsmlla;
					task_trsm_ll->cl_arg_size = sizeof(*utrsmlla);

					utrsmlla->i = i;
					utrsmlla->k = slice;
					utrsmlla->nblocks = args->nblocks;
					utrsmlla->dataA = args->dataA;

					task_trsm_ll->handles[0] = starpu_data_get_sub_data(args->dataA, 2, utrsmlla->i, utrsmlla->i);
					task_trsm_ll->handles[1] = starpu_data_get_sub_data(args->dataA, 2, utrsmlla->k, utrsmlla->i);

					if (!no_prio && (slice == i +1))
						task_trsm_ll->priority = STARPU_MAX_PRIO;

					debug("ugetrf %d start utrsmll %d %d\n", i, i, slice);
					ret = starpu_task_submit(task_trsm_ll);
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
					cl_args *utrsmrua = malloc(sizeof(cl_args));

					struct starpu_task *task_trsm_ru = starpu_task_create();
					task_trsm_ru->callback_func = dw_callback_v2_codelet_update_trsm_ru;
					task_trsm_ru->callback_arg = utrsmrua;
					task_trsm_ru->cl = &cl_trsm_ru;
					task_trsm_ru->cl_arg = utrsmrua;
					task_trsm_ru->cl_arg_size = sizeof(*utrsmrua);

					utrsmrua->i = i;
					utrsmrua->k = slice;
					utrsmrua->nblocks = args->nblocks;
					utrsmrua->dataA = args->dataA;

					task_trsm_ru->handles[0] = starpu_data_get_sub_data(args->dataA, 2, utrsmrua->i, utrsmrua->i);
					task_trsm_ru->handles[1] = starpu_data_get_sub_data(args->dataA, 2, utrsmrua->i, utrsmrua->k);

					if (!no_prio && (slice == i +1))
						task_trsm_ru->priority = STARPU_MAX_PRIO;

					debug("ugetrf %d start utrsmru %d %d\n", i, i, slice);
					ret = starpu_task_submit(task_trsm_ru);
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


void dw_callback_codelet_update_getrf(void *argcb)
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

			/* update slice from utrsmll */
			cl_args *utrsmlla = malloc(sizeof(cl_args));

			/* update slice from utrsmru */
			cl_args *utrsmrua = malloc(sizeof(cl_args));

			struct starpu_task *task_trsm_ll = starpu_task_create();
			task_trsm_ll->callback_func = dw_callback_codelet_update_trsm_ll_21;
			task_trsm_ll->callback_arg = utrsmlla;
			task_trsm_ll->cl = &cl_trsm_ll;
			task_trsm_ll->cl_arg = utrsmlla;
			task_trsm_ll->cl_arg_size = sizeof(*utrsmlla);

			struct starpu_task *task_trsm_ru = starpu_task_create();
			task_trsm_ru->callback_func = dw_callback_codelet_update_trsm_ll_21;
			task_trsm_ru->callback_arg = utrsmrua;
			task_trsm_ru->cl = &cl_trsm_ru;
			task_trsm_ru->cl_arg = utrsmrua;
			task_trsm_ru->cl_arg_size = sizeof(*utrsmrua);

			utrsmlla->i = args->i;
			utrsmlla->k = slice;
			utrsmlla->nblocks = args->nblocks;
			utrsmlla->dataA = args->dataA;
			utrsmlla->remaining = remaining;

			utrsmrua->i = args->i;
			utrsmrua->k = slice;
			utrsmrua->nblocks = args->nblocks;
			utrsmrua->dataA = args->dataA;
			utrsmrua->remaining = remaining;

			task_trsm_ll->handles[0] = starpu_data_get_sub_data(args->dataA, 2, utrsmlla->i, utrsmlla->i);
			task_trsm_ll->handles[1] = starpu_data_get_sub_data(args->dataA, 2, utrsmlla->k, utrsmlla->i);

			task_trsm_ru->handles[0] = starpu_data_get_sub_data(args->dataA, 2, utrsmrua->i, utrsmrua->i);
			task_trsm_ru->handles[1] = starpu_data_get_sub_data(args->dataA, 2, utrsmrua->i, utrsmrua->k);

			ret = starpu_task_submit(task_trsm_ll);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			ret = starpu_task_submit(task_trsm_ru);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		free(remaining);
	}
}


void dw_callback_codelet_update_gemm(void *argcb)
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
		cl_args *ugetrfarg = malloc(sizeof(cl_args));

		struct starpu_task *task = starpu_task_create();
		task->callback_func = dw_callback_codelet_update_getrf;
		task->callback_arg = ugetrfarg;
		task->cl = &cl_getrf;
		task->cl_arg = ugetrfarg;
		task->cl_arg_size = sizeof(*ugetrfarg);

		task->handles[0] = starpu_data_get_sub_data(args->dataA, 2, args->k + 1, args->k + 1);

		ugetrfarg->dataA = args->dataA;
		ugetrfarg->i = args->k + 1;
		ugetrfarg->nblocks = args->nblocks;

		/* schedule the codelet */
		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	free(args);
}

void dw_callback_codelet_update_trsm_ll_21(void *argcb)
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
				cl_args *ugemma = malloc(sizeof(cl_args));

				struct starpu_task *task_gemm = starpu_task_create();
				task_gemm->callback_func = dw_callback_codelet_update_gemm;
				task_gemm->callback_arg = ugemma;
				task_gemm->cl = &cl_gemm;
				task_gemm->cl_arg = ugemma;
				task_gemm->cl_arg_size = sizeof(*ugemma);

				ugemma->k = i;
				ugemma->i = slicex;
				ugemma->j = slicey;
				ugemma->dataA = args->dataA;
				ugemma->nblocks = nblocks;
				ugemma->remaining = remaining_tasks;

				task_gemm->handles[0] = starpu_data_get_sub_data(args->dataA, 2, ugemma->i, ugemma->k);
				task_gemm->handles[1] = starpu_data_get_sub_data(args->dataA, 2, ugemma->k, ugemma->j);
				task_gemm->handles[2] = starpu_data_get_sub_data(args->dataA, 2, ugemma->i, ugemma->j);

				/* schedule that codelet */
				ret = starpu_task_submit(task_gemm);
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
	task->callback_func = dw_callback_codelet_update_getrf;
	task->callback_arg = args;
	task->cl = &cl_getrf;
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

	PRINTF("# size\tms\tGFlop/s\n");
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
	task->callback_func = dw_callback_v2_codelet_update_getrf;
	task->callback_arg = args;
	task->cl = &cl_getrf;
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

	PRINTF("# size\tms\tGFlop/s\n");
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
	char * symbol_getrf = "lu_model_getrf_atlas";
	char * symbol_trsm_ll = "lu_model_trsm_ll_atlas";
	char * symbol_trsm_ru = "lu_model_trsm_ru_atlas";
	char * symbol_gemm = "lu_model_gemm_atlas";
#elif defined(STARPU_GOTO)
	char * symbol_getrf = "lu_model_getrf_goto";
	char * symbol_trsm_ll = "lu_model_trsm_ll_goto";
	char * symbol_trsm_ru = "lu_model_trsm_ru_goto";
	char * symbol_gemm = "lu_model_gemm_goto";
#elif defined(STARPU_OPENBLAS)
	char * symbol_getrf = "lu_model_getrf_openblas";
	char * symbol_trsm_ll = "lu_model_trsm_ll_openblas";
	char * symbol_trsm_ru = "lu_model_trsm_ru_openblas";
	char * symbol_gemm = "lu_model_gemm_openblas";
#else
	char * symbol_getrf = "lu_model_getrf";
	char * symbol_trsm_ll = "lu_model_trsm_ll";
	char * symbol_trsm_ru = "lu_model_trsm_ru";
	char * symbol_gemm = "lu_model_gemm";
#endif
	initialize_lu_kernels_model(&model_getrf,symbol_getrf,task_getrf_cost,task_getrf_cost_cpu,task_getrf_cost_cuda);
	initialize_lu_kernels_model(&model_trsm_ll,symbol_trsm_ll,task_trsm_ll_cost,task_trsm_ll_cost_cpu,task_trsm_ll_cost_cuda);
	initialize_lu_kernels_model(&model_trsm_ru,symbol_trsm_ru,task_trsm_ru_cost,task_trsm_ru_cost_cpu,task_trsm_ru_cost_cuda);
	initialize_lu_kernels_model(&model_gemm,symbol_gemm,task_gemm_cost,task_gemm_cost_cpu,task_gemm_cost_cuda);

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

void free_system(float *A, float *B, unsigned dim, unsigned pinned)
{
	if (pinned)
	{
		starpu_free_noflag(A, (size_t)dim*dim*sizeof(float));
		starpu_free_noflag(B, (size_t)dim*sizeof(float));
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
