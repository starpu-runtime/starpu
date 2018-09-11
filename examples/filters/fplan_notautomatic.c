/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018                                     CNRS
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define NX    9
#define PARTS 3

struct starpu_codelet task_codelet;

// CPU implementations
void task_cpu(void *descr[], void *args)
{
	int *values = (int*)STARPU_VECTOR_GET_PTR(descr[0]);
	int nx = STARPU_VECTOR_GET_NX(descr[0]);
	int i, add;
	char message[10000];
	int cur = 0;

	starpu_codelet_unpack_args(args, &add);

	cur += snprintf(&message[cur], 10000-cur, "Values ");
	for(i=0 ; i<nx ; i++)
	{
		values[i] += add;
		cur += snprintf(&message[cur], 10000-cur, "%d ", values[i]);
	}
	FPRINTF(stderr, "%s\n", message);
}

void split_cpu(void *descr[], void *args)
{
	(void)descr;
	//	starpu_data_handle_t data_handle = starpu_data_lookup((void*)STARPU_VECTOR_GET_PTR(descr[0]));

	starpu_data_handle_t value_handle, sub_handles[PARTS];
	starpu_codelet_unpack_args(args, &value_handle, &sub_handles);

	FPRINTF(stderr, "Partition for handle %p into handles %p %p and %p\n", value_handle, sub_handles[0], sub_handles[1], sub_handles[2]);

	starpu_data_partition_submit_sequential_consistency(value_handle, PARTS, sub_handles, 0);
}

void supertask_cpu(void *descr[], void *args)
{
	(void)descr;
	//	starpu_data_handle_t data_handle = starpu_data_lookup((void*)STARPU_VECTOR_GET_PTR(descr[0]));
	starpu_data_handle_t sub_handles[PARTS];
	int add;

	starpu_codelet_unpack_args(args, &sub_handles, &add);

	FPRINTF(stderr, "Submitting tasks on %d subdata (add %d)\n", PARTS, add);

	int i;
	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&task_codelet,
					     STARPU_RW, sub_handles[i],
					     STARPU_VALUE, &add, sizeof(add),
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

void merge_cpu(void *descr[], void *args)
{
	(void)descr;
	//	starpu_data_handle_t value_handle = starpu_data_lookup((void*)STARPU_VECTOR_GET_PTR(descr[0]));

	starpu_data_handle_t value_handle, sub_handles[PARTS];
	starpu_codelet_unpack_args(args, &value_handle, &sub_handles);

	FPRINTF(stderr, "Unpartition for handle %p from handles %p %p and %p\n", value_handle, sub_handles[0], sub_handles[1], sub_handles[2]);

	starpu_data_unpartition_submit_sequential_consistency(value_handle, PARTS, sub_handles, STARPU_MAIN_RAM, 0);
}


// Codelets
struct starpu_codelet task_codelet =
{
	.cpu_funcs = {task_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "task_codelet"
};

struct starpu_codelet supertask_codelet =
{
	.cpu_funcs = {supertask_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "supertask_codelet"
};

struct starpu_codelet split_codelet =
{
	.cpu_funcs = {split_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "split_codelet"
};

struct starpu_codelet merge_codelet =
{
	.cpu_funcs = {merge_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "merge_codelet"
};

int main(void)
{
	int ret, i;
	int values[NX];
	int check[NX];
	int add=1;
	starpu_data_handle_t value_handle;
	starpu_data_handle_t sub_handles[PARTS];

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_shutdown();
		return 77;
	}

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = PARTS
	};

	values[NX-1] = 2;
	for(i=NX-2 ; i>= 0 ; i--) values[i] = values[i+1] * 2;
	for(i=0 ; i<NX ; i++) check[i] = values[i] + (4 * add);

	starpu_vector_data_register(&value_handle, STARPU_MAIN_RAM, (uintptr_t)&values[0], NX, sizeof(values[0]));
	starpu_data_partition_plan(value_handle, &f, sub_handles);

	// tell StarPU not to partition data, the application will decide itself when to do it
	starpu_data_partition_not_automatic(value_handle);
	for(i=0 ; i<PARTS ; i++)
		starpu_data_partition_not_automatic(sub_handles[i]);

	// insert a task on the whole data
	ret = starpu_task_insert(&task_codelet, STARPU_RW, value_handle,
				 STARPU_VALUE, &add, sizeof(add),
				 STARPU_NAME, "task_1", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// insert a task to split the data
	ret = starpu_task_insert(&split_codelet, STARPU_RW, value_handle,
				 STARPU_VALUE, &value_handle, sizeof(starpu_data_handle_t),
				 STARPU_VALUE, sub_handles, PARTS*sizeof(starpu_data_handle_t),
				 STARPU_NAME, "split", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// insert a task that will work on the subdata
	ret = starpu_task_insert(&supertask_codelet, STARPU_RW, value_handle,
				 STARPU_VALUE, sub_handles, PARTS*sizeof(starpu_data_handle_t),
				 STARPU_VALUE, &add, sizeof(add),
				 STARPU_NAME, "supertask_1", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// insert another task that will work on the subdata
	ret = starpu_task_insert(&supertask_codelet, STARPU_RW, value_handle,
				 STARPU_VALUE, sub_handles, PARTS*sizeof(starpu_data_handle_t),
				 STARPU_VALUE, &add, sizeof(add),
				 STARPU_NAME, "supertask_2", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// insert a task to merge the data
	ret = starpu_task_insert(&merge_codelet, STARPU_RW, value_handle,
				 STARPU_VALUE, &value_handle, sizeof(starpu_data_handle_t),
				 STARPU_VALUE, sub_handles, PARTS*sizeof(starpu_data_handle_t),
				 STARPU_NAME, "merge", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// insert a task that will work on the whole data
	ret = starpu_task_insert(&task_codelet, STARPU_RW, value_handle,
				 STARPU_VALUE, &add, sizeof(add),
				 STARPU_NAME, "task_2", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();
	starpu_data_partition_clean(value_handle, PARTS, sub_handles);
	starpu_data_unregister(value_handle);

	FPRINTF(stderr, "Values : ");
	for(i=0 ; i<NX ; i++)
	{
		FPRINTF(stderr, "%d ", values[i]);
	}
	FPRINTF(stderr, "\n");
	for(i=0 ; i<NX ; i++)
	{
		if (values[i] != check[i])
		{
			FPRINTF(stderr, "Incorrect value for %d. %d != %d\n", i, values[i], check[i]);
			ret = 1;
		}
	}

	starpu_shutdown();

	return ret;
}
