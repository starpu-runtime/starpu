/*
 * StarPU
 * Copyright (C) INRIA 2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR in PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include <starpu.h>

#include "starpufft.h"

#define _FFTW_FLAGS FFTW_ESTIMATE

enum steps {
	START, SHUFFLED_1, FFT_1, SHUFFLED_2, FFT_2, SHUFFLED_3, END
};

// TODO: Z2Z, D2Z, Z2D
enum type {
	R2C,
	C2R,
	C2C
};

static unsigned task_per_worker[STARPU_NMAXWORKERS];
static unsigned samples_per_worker[STARPU_NMAXWORKERS];
static struct timeval start, submit_tasks, do_tasks, tasks_done, gather, end;

/*
 *
 *	The actual kernels
 *
 */

struct STARPUFFT(plan) {
	int number;	/* uniquely identifies the plan, for starpu tags */

	int *n;
	int *n1;
	int *n2;
	int totsize;
	int totsize1;
	int totsize2;
	int dim;
	enum type type;
	int sign;

	STARPUFFT(complex) *roots[2];
	starpu_data_handle roots_handle[2];

	/* Synchronization for termination */
	unsigned todo;
	pthread_mutex_t mutex;
	pthread_cond_t cond;

	struct {
#ifdef USE_CUDA
		cufftHandle plan_cuda;
		_cufftComplex *gpu_in;
		STARPUFFT(complex) *local_in;
		int initialized;
#endif
#ifdef HAVE_FFTW
		_fftw_plan plan_cpu;
		_fftw_complex *in;
		_fftw_complex *out;
#endif
	} plans[STARPU_NMAXWORKERS];

#ifdef HAVE_FFTW
	_fftw_plan plan_gather;
#endif

	STARPUFFT(complex) *in;
	STARPUFFT(complex) *split_in, *split_out;
	STARPUFFT(complex) *output;

	starpu_data_handle *in_handle;
	starpu_data_handle *out_handle;
	struct starpu_task **tasks;
	struct STARPUFFT(args) *args;
};

struct STARPUFFT(args) {
	struct STARPUFFT(plan) *plan;
	int i, j, *iv;
};

void
STARPUFFT(callback)(void *_plan)
{
	STARPUFFT(plan) plan = _plan;

	int workerid = starpu_get_worker_id();

	/* do some accounting */
	task_per_worker[workerid]++;
	samples_per_worker[workerid] += plan->totsize2;

	if (STARPU_ATOMIC_ADD(&plan->todo, -1) == 0)
	{
		pthread_mutex_lock(&plan->mutex);
		pthread_cond_signal(&plan->cond);
		pthread_mutex_unlock(&plan->mutex);
	}
}

static void
check_dims(STARPUFFT(plan) plan)
{
	int dim;
	for (dim = 0; dim < plan->dim; dim++)
		if (plan->n[dim] & (plan->n[dim]-1)) {
			fprintf(stderr,"can't cope with non-power-of-2\n");
			STARPU_ASSERT(0);
		}
}

static void
compute_roots(STARPUFFT(plan) plan)
{
	int dim, k;

	/* Compute the n-roots and m-roots of unity for twiddling */
	for (dim = 0; dim < plan->dim; dim++) {
		STARPUFFT(complex) exp = (plan->sign * 2. * 4.*atan(1.)) * _Complex_I / (STARPUFFT(complex)) plan->n[dim];
		plan->roots[dim] = malloc(plan->n[dim] * sizeof(**plan->roots));
		for (k = 0; k < plan->n[dim]; k++)
			plan->roots[dim][k] = cexp(exp*k);
		starpu_register_vector_data(&plan->roots_handle[dim], 0, (uintptr_t) plan->roots[dim], plan->n[dim], sizeof(**plan->roots));
	}
}

#include "starpufftx1d.c"
#include "starpufftx2d.c"

void
STARPUFFT(execute)(STARPUFFT(plan) plan, void *_in, void *_out)
{
	gettimeofday(&start, NULL);
	memset(task_per_worker, 0, sizeof(task_per_worker));
	memset(samples_per_worker, 0, sizeof(task_per_worker));

	plan->in = _in;

	switch (plan->dim) {
		case 1: {
			switch (plan->type) {
			case C2C:
				STARPUFFT(execute1dC2C)(plan, _in, _out);
				break;
			default:
				STARPU_ASSERT(0);
				break;
			}
			break;
		}
		case 2:
			STARPUFFT(execute2dC2C)(plan, _in, _out);
			break;
		default:
			STARPU_ASSERT(0);
			break;
	}

	gettimeofday(&end, NULL);
}

void
STARPUFFT(destroy_plan)(STARPUFFT(plan) plan)
{
	int workerid, dim, i;

	for (workerid = 0; workerid < starpu_get_worker_count(); workerid++) {
		switch (starpu_get_worker_type(workerid)) {
		case STARPU_CORE_WORKER:
#ifdef HAVE_FFTW
			_FFTW(free)(plan->plans[workerid].in);
			_FFTW(free)(plan->plans[workerid].out);
			_FFTW(destroy_plan)(plan->plans[workerid].plan_cpu);
#endif
			break;
		case STARPU_CUDA_WORKER:
#ifdef USE_CUDA
			/* FIXME: Can't deallocate */
#endif
			break;
		default:
			STARPU_ASSERT(0);
			break;
		}
	}
	for (i = 0; i < plan->totsize1; i++) {
		starpu_delete_data(plan->in_handle[i]);
		starpu_delete_data(plan->out_handle[i]);
		free(plan->tasks[i]);
	}
	free(plan->in_handle);
	free(plan->out_handle);
	free(plan->tasks);
	free(plan->args);
	for (dim = 0; dim < plan->dim; dim++) {
		starpu_delete_data(plan->roots_handle[dim]);
		free(plan->roots[dim]);
	}
	free(plan->n);
	free(plan->n1);
	free(plan->n2);
	STARPUFFT(free)(plan->split_in);
	STARPUFFT(free)(plan->split_out);
	STARPUFFT(free)(plan->output);
#ifdef HAVE_FFTW
	_FFTW(destroy_plan)(plan->plan_gather);
#endif
	free(plan);
}

void *
STARPUFFT(malloc)(size_t n)
{
#ifdef USE_CUDA
	void *res;
	starpu_malloc_pinned_if_possible(&res, n);
	return res;
#else
#  ifdef HAVE_FFTW
	return _FFTW(malloc)(n);
#  else
	return malloc(n);
#  endif
#endif
}

void
STARPUFFT(free)(void *p)
{
#ifdef USE_CUDA
	// TODO: FIXME
#else
#  ifdef HAVE_FFTW
	_FFTW(free)(p);
#  else
	free(p);
#  endif
#endif
}

void
STARPUFFT(showstats)(FILE *out)
{
	int worker;
	unsigned total;

#define TIMING(begin,end) (double)((end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec))
#define MSTIMING(begin,end) (TIMING(begin,end)/1000.)
	double paratiming = TIMING(start,do_tasks);
	fprintf(out, "Tasks submission took %2.2f ms\n", MSTIMING(start,submit_tasks));
	fprintf(out, "Tasks termination took %2.2f ms\n", MSTIMING(submit_tasks,do_tasks));
	fprintf(out, "Tasks cleanup took %2.2f ms\n", MSTIMING(do_tasks,tasks_done));
	fprintf(out, "Gather took %2.2f ms\n", MSTIMING(tasks_done,gather));
	fprintf(out, "Finalization took %2.2f ms\n", MSTIMING(gather,end));

	fprintf(out, "Total %2.2f ms\n", MSTIMING(start,end));

	for (worker = 0, total = 0; worker < STARPU_NMAXWORKERS; worker++)
		total += task_per_worker[worker];

	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		if (task_per_worker[worker])
		{
			char name[32];
			starpu_get_worker_name(worker, name, 32);

			unsigned long bytes = sizeof(STARPUFFT(complex))*samples_per_worker[worker];

			fprintf(stderr, "\t%s -> %2.2f MB\t%2.2f\tMB/s\t%u %2.2f %%\n", name, (1.0*bytes)/(1024*1024), bytes/paratiming, task_per_worker[worker], (100.0*task_per_worker[worker])/total);
		}
	}
}
