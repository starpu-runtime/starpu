/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019-2019  Gwenole Lucas
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

#include <starpu.h>

#define X 20
#define SLICES 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void func_cpu(void *descr[], void *_args)
{
	(void) _args;
	size_t x;
	size_t nx = STARPU_VECTOR_GET_NX(descr[0]);
	int *v = (int *)STARPU_VECTOR_GET_PTR(descr[0]);
	int sum=0;

	for(x=0 ; x<nx ; x++)
		sum += v[x];
	for(x=0 ; x<nx ; x++)
		v[x] = sum/nx;
}

struct starpu_codelet vector_cl =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

struct starpu_data_filter f =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = SLICES
};

int vector_no_recursive_task()
{
	int *vector;
	starpu_data_handle_t vhandle;
	int ret, x, loop;

	vector = malloc(X*sizeof(vector[0]));
	for(x=0 ; x<X ; x++)
		vector[x]=x+1;

	FPRINTF(stderr, "Vector in\n");
	for(x=0 ; x<X ; x++)
	{
		FPRINTF(stderr, "%3d", vector[x]);
		FPRINTF(stderr, " ");
	}
	FPRINTF(stderr, "\n");

	starpu_vector_data_register(&vhandle, 0, (uintptr_t)vector, X, sizeof(vector[0]));
	starpu_data_map_filters(vhandle, 1, &f);

	for (x = 0; x < SLICES; x++)
	{
		ret = starpu_task_insert(&vector_cl,
					 STARPU_RW, starpu_data_get_sub_data(vhandle, 1, x),
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

enodev:
	starpu_data_unpartition(vhandle, STARPU_MAIN_RAM);
	starpu_data_unregister(vhandle);

	if (ret != -ENODEV)
	{
		FPRINTF(stderr, "Vector out\n");
		for(x=0 ; x<X ; x++)
		{
			FPRINTF(stderr, "%3d", vector[x]);
			FPRINTF(stderr, " ");
		}
		FPRINTF(stderr, "\n");
	}

	free(vector);
	return (ret == -ENODEV) ? 77 : 0;
}

int is_recursive_task(struct starpu_task *t, void *arg, void **b)
{
	(void)arg;
	(void)b;
	starpu_data_handle_t *handles = STARPU_TASK_GET_HANDLES(t);
	return (starpu_data_partition_get_nplans(handles[0]) > 0);
}

starpu_data_handle_t sub_handles[SLICES];
starpu_data_handle_t sub_sub_handles[SLICES];

void recursive_task_gen_dag(struct starpu_task *t, void *arg, void **b);
struct starpu_codelet recursive_task_codelet =
{
	.cpu_funcs = {func_cpu},
	.recursive_task_func = is_recursive_task,
	.recursive_task_gen_dag_func = recursive_task_gen_dag,
	.nbuffers = 1
};

void recursive_task_gen_dag(struct starpu_task *t, void *arg, void **b)
{
	int i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<SLICES ; i++)
	{
		int ret = starpu_task_insert(&recursive_task_codelet,
					     STARPU_RW, subdata[i],
					     STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_sub_handles,
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

int vector_recursive_task()
{
	int *vector;
	starpu_data_handle_t vhandle;
	int ret, x, loop;

	vector = malloc(X*sizeof(vector[0]));
	for(x=0 ; x<X ; x++)
		vector[x]=x+1;

	FPRINTF(stderr, "Initial vector\n");
	for(x=0 ; x<X ; x++)
	{
		FPRINTF(stderr, "%3d", vector[x]);
		FPRINTF(stderr, " ");
	}
	FPRINTF(stderr, "\n");

	starpu_vector_data_register(&vhandle, 0, (uintptr_t)vector, X, sizeof(vector[0]));
	starpu_data_partition_plan(vhandle, &f, sub_handles);
	starpu_data_partition_plan(sub_handles[0], &f, sub_sub_handles);

	ret = starpu_task_insert(&recursive_task_codelet,
				 STARPU_RW, vhandle,
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

enodev:
	starpu_task_wait_for_all();
	starpu_data_partition_clean(sub_handles[0], SLICES, sub_sub_handles);
	starpu_data_partition_clean(vhandle, SLICES, sub_handles);
	starpu_data_unregister(vhandle);

	if (ret != -ENODEV)
	{
		FPRINTF(stderr, "Vector reloaded\n");
		for(x=0 ; x<X ; x++)
		{
			FPRINTF(stderr, "%3d", vector[x]);
			FPRINTF(stderr, " ");
		}
		FPRINTF(stderr, "\n");
	}

	free(vector);
	return (ret == -ENODEV) ? 77 : 0;
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_shutdown();
		return 77;
	}

	ret = vector_no_recursive_task();
	if (ret == 77)
	{
		starpu_shutdown();
		return 77;
	}

	ret = vector_recursive_task();
	if (ret == 77)
	{
		starpu_shutdown();
		return 77;
	}

	starpu_shutdown();
	return 0;
}
