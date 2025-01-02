/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This shows how to implement an spmd parallel StarPU task: scal_cpu_func is
 * called in parallel over several cores, and has to split the work accordingly.
 * This is a mere vector scaling example.
 */

/* gcc build:
 *
 *  gcc -O2 -g vector_scal.c -o vector_scal $(pkg-config --cflags starpu-1.0) $(pkg-config --libs starpu-1.0)
 *
 */

#include <starpu.h>
#include <stdio.h>
#include <limits.h>

#define MIN(a,b)        ((a)<(b)?(a):(b))

#define	NX	204800
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#ifdef STARPU_QUICK_CHECK
#define ITER 10
#else
#define ITER 100
#endif

static int get_first_element_rank(int nel, int rank, int nb_workers)
{
	if(rank == 0)
		return 0;

	/* We get the  number of bigger parts which stand before the part */
	int nb_big_parts = MIN(nel % nb_workers, rank);

	return nb_big_parts * (nel / nb_workers + 1) + (rank - nb_big_parts) * (nel / nb_workers);
}

void scal_cpu_func(void *buffers[], void *_args)
{
	int i;
	float *factor = _args, f = *factor;
	struct starpu_vector_interface *vector = buffers[0];
	int n = STARPU_VECTOR_GET_NX(vector);
	float *val = (float *)STARPU_VECTOR_GET_PTR(vector);

	int nb_workers = starpu_combined_worker_get_size();
	int rank = starpu_combined_worker_get_rank();

	if (rank == 0)
		FPRINTF(stderr, "running task with %d CPUs.\n", starpu_combined_worker_get_size());

	/* We add 1 to the (nel_total % nb_workers) first workers, thus we get an evenly split data. */
	int nel_worker = (n / nb_workers) + ((rank < (n % nb_workers)) ? 1 : 0);

	int begin = get_first_element_rank(n, rank, nb_workers);


	for (i = 0; i < nel_worker; i++)
	{
		rank = i + begin;

		float v = val[rank];
		int j;
		for (j = 0; j < 100; j++)
			v = v * f;
		val[rank] = v;
	}
}

static struct starpu_perfmodel vector_scal_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "vector_scal_parallel"
};

static struct starpu_codelet cl =
{
	.modes = { STARPU_RW },
	.type = STARPU_SPMD,
	.max_parallelism = INT_MAX,
	.cpu_funcs = {scal_cpu_func},
	.cpu_funcs_name = {"scal_cpu_func"},
	.nbuffers = 1,
	.model = &vector_scal_model,
};

int main(void)
{
	struct starpu_conf conf;
	float *vector;
	unsigned i;
	int ret;

	starpu_conf_init(&conf);
	conf.single_combined_worker = 1;
	conf.sched_policy_name = "pheft";

	{
		ret = starpu_init(NULL);
		if (ret == -ENODEV) return 77;
		conf.ncpus = starpu_cpu_worker_get_count();
		conf.ncpus /= 2;
		starpu_shutdown();
	}


	ret = starpu_init(&conf);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&vector, NX*sizeof(float));
	for (i = 0; i < NX; i++)
		vector[i] = (i+1.0f);

	FPRINTF(stderr, "BEFORE: First element was %f\n", vector[0]);
	FPRINTF(stderr, "BEFORE: Last element was %f\n", vector[NX-1]);

	starpu_data_handle_t vector_handle;
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

	float factor = 1.001;

	for (i = 0; i < ITER; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;

		task->handles[0] = vector_handle;
		task->cl_arg = &factor;
		task->cl_arg_size = sizeof(factor);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV)
		{
		     ret = 77;
		     break;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(vector_handle);

	FPRINTF(stderr, "AFTER: First element is %f\n", vector[0]);
	FPRINTF(stderr, "AFTER: Last element is %f\n", vector[NX-1]);

	starpu_free_noflag(vector, NX*sizeof(float));

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	return ret;
}
