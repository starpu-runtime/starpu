/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef PARTS
#define PARTS 5
#endif
#ifndef SIZE
#define SIZE 25
#endif

#define check_recursive_task(x) x*=2
#define check_task(x) x+=10

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

struct starpu_data_filter f =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = PARTS
};

void print_vector(int *v, size_t nx, const char *label)
{
	char message[100000];
	int cur=0;
	size_t i;

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
	size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
	size_t i;

	for(i=0 ; i<nx ; i++)
		v[i] *= 2;
	char str[35];
	sprintf(str, "subtask_%d", (int)(uintptr_t) arg);
	print_vector(v, nx, str);
}

extern void sub_data_cuda_func(void *buffers[], void *arg);

struct starpu_codelet sub_data_codelet =
{
	.cpu_funcs = {sub_data_func},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {sub_data_cuda_func},
#endif
	.nbuffers = 1,
	.name = "task_cl",
	.model = &starpu_perfmodel_nop
};

void sub_data_RO_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
	print_vector(v, nx, "subtaskRO");
}

extern void sub_data_RO_cuda_func(void *buffers[], void *arg);

struct starpu_codelet sub_data_RO_codelet =
{
	.cpu_funcs = {sub_data_RO_func},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {sub_data_RO_cuda_func},
#endif
	.nbuffers = 1,
	.name = "subtaskRO_cl",
	.model = &starpu_perfmodel_nop
};

void recursive_task_func(void *buffers[], void *arg)
{
	assert(0);
	return;
}

int is_recursive_task(struct starpu_task *t, void *arg)
{
	(void)arg;
	(void)t;
	//starpu_data_handle_t *handles = STARPU_TASK_GET_HANDLES(t);
	//	return starpu_data_get_nb_children_async(handles[0]) > 0;
	return 1;
}

// Function which change when called a second time
int is_recursive_task_only_second_time(struct starpu_task *t, void *arg)
{
	assert(arg);
	int * v = (int*) arg;
	if (*v)
		return 1;
	*v = 1;
	return 0;
}

void recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Hello i am a recursive task\n");
	int i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		char *str = calloc(25, sizeof(char));
		sprintf(str, "sub_data_%d", i);
		int ret = starpu_task_insert(&sub_data_codelet,
					     STARPU_CL_ARGS_NFREE, i, sizeof(int),
					     STARPU_RW, subdata[i],
					     STARPU_RECURSIVE_TASK_PARENT, t,
					     STARPU_NAME, str,
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

struct starpu_codelet recursive_task_second_call_codelet =
{
	.cpu_funcs = {recursive_task_func},
	.recursive_task_func = is_recursive_task_only_second_time,
	.recursive_task_gen_dag_func = recursive_task_gen_dag,
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop
};

struct starpu_codelet recursive_task_codelet =
{
	.cpu_funcs = {recursive_task_func},
	.cuda_funcs = {recursive_task_func},
	.recursive_task_func = is_recursive_task,
	.recursive_task_gen_dag_func = recursive_task_gen_dag,
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop
};

void recursive_taskRO_gen_dag(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Hello i am a recursive_task\n");
	int i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_RO_codelet,
					     STARPU_R, subdata[i],
					     STARPU_NAME, "sub_data_RO",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

struct starpu_codelet recursive_taskRO_codelet =
{
	.cpu_funcs = {recursive_task_func},
	.recursive_task_func = is_recursive_task,
	.recursive_task_gen_dag_func = recursive_taskRO_gen_dag,
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop
};

void task_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
	size_t i;

	print_vector(v, nx, "task");
	for(i=0 ; i<nx ; i++)
	{
		v[i] += 10;
	}
}

extern void task_cuda_func(void *buffers[], void *arg);

struct starpu_codelet task_codelet =
{
	.cpu_funcs = {task_func},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {task_cuda_func},
#endif
	.nbuffers = 1
};

void task_RO_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
	print_vector(v, nx, "taskRO");
}

extern void task_RO_cuda_func(void *buffers[], void *arg);

struct starpu_codelet taskRO_codelet =
{
	.cpu_funcs = {task_RO_func},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {task_RO_cuda_func},
#endif
	.nbuffers = 1
};
