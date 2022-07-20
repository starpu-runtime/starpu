/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Gwenole Lucas
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

#ifndef PARTS
#define PARTS 5
#endif
#ifndef SIZE
#define SIZE  25
#endif

#define check_bubble(x) x*=2
#define check_task(x) x+=10

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

struct starpu_data_filter f =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = PARTS
};

void print_vector(int *v, int nx, const char *label)
{
	char message[100000];
	int cur=0;
	int i;

	cur += snprintf(&message[cur], 100000 - cur, "%s : ", label);
	for (i=0; i<nx-1; i++)
	{
		cur += snprintf(&message[cur], 100000 - cur, "%3d,", v[i]);
	}
	snprintf(&message[cur], 100000 - cur, "%3d\n", v[nx-1]);
	FPRINTF(stderr, message);
}

void sub_data_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;

	for(i=0 ; i<nx ; i++)
		v[i] *= 2;
	struct starpu_task *task = starpu_task_get_current();
	print_vector(v, nx, starpu_task_get_name(task));
}

struct starpu_codelet sub_data_codelet =
{
	.cpu_funcs = {sub_data_func},
	.nbuffers = 1,
	.name = "sub_data_cl",
	.model = &starpu_perfmodel_nop
};

void bubble_func(void *buffers[], void *arg)
{
	assert(0);
	return;
}

int is_bubble(struct starpu_task *t, void *arg)
{
	(void)arg;
	(void)t;
	//starpu_data_handle_t *handles = STARPU_TASK_GET_HANDLES(t);
	//	return starpu_data_get_nb_children_async(handles[0]) > 0;
	return 1;
}

void bubble_gen_dag(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Hello i am a bubble\n");
	int i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_codelet,
					     STARPU_RW, subdata[i],
					     STARPU_NAME, "sub_data",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

struct starpu_codelet bubble_codelet =
{
	.cpu_funcs = {bubble_func},
	.bubble_func = is_bubble,
	.bubble_gen_dag_func = bubble_gen_dag,
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop
};

void task_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;

	print_vector(v, nx, "task");
	for(i=0 ; i<nx ; i++)
	{
		v[i] += 10;
	}
}

struct starpu_codelet task_codelet =
{
	.cpu_funcs = {task_func},
	.nbuffers = 1
};

