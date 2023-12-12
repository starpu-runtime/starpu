/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "starpu.h"

#define NPARTS 4
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void init_cpu(void* buffers[], void *args)
{
	double *v = (double*)STARPU_VECTOR_GET_PTR(buffers[0]);
	unsigned nx = STARPU_VECTOR_GET_NX(buffers[0]);
	unsigned i;
	for (i=0; i<nx; ++i) v[i] = 0;
}

int main(int argc, char** argv)
{
	return 77;
	int i, ret;
	starpu_data_handle_t parent;
	starpu_data_handle_t parent_sync;
	starpu_data_handle_t children[NPARTS];

	struct starpu_conf conf;
	starpu_conf_init(&conf);

#ifdef STARPU_DEVEL
#warning FIXME: this example does not support Master-Slave
#endif
	conf.nmpi_ms = 0;
	conf.ntcpip_ms = 0;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_codelet init_cl =
	{
		.cpu_funcs = {init_cpu},
		.cpu_funcs_name = {"init_cpu"},
		.nbuffers = 1,
		.modes = {STARPU_W},
		.name = "init_cl"
	};

	struct starpu_data_filter f_vert =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = NPARTS
	};

	// Asynchronous partitioning
	starpu_vector_data_register(&parent, -1, 0, NPARTS*4, sizeof(double));
	starpu_data_partition_plan(parent, &f_vert, children);

	for(i=0 ; i<NPARTS; i++)
	{
		ret = starpu_task_insert(&init_cl, STARPU_W, children[i], 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_data_partition_clean_node(parent, f_vert.nchildren, children, -1);
	starpu_data_unregister(parent);

	// Synchronous partitioning
	starpu_vector_data_register(&parent_sync, -1, 0, NPARTS*4, sizeof(double));
	starpu_data_partition(parent_sync, &f_vert);

	for (i=0; i<starpu_data_get_nb_children(parent_sync); i++)
	{
		ret = starpu_task_insert(&init_cl, STARPU_W, starpu_data_get_sub_data(parent_sync, 1, i), 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_data_unpartition(parent_sync, -1);
	starpu_data_unregister(parent_sync);

	starpu_shutdown();
	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;

}
