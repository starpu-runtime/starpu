/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

void specific_kernel(void *descr[], void *arg)
{
	(void)arg;
	int i;
	int node = starpu_task_get_current_data_node(0);

	int num = STARPU_TASK_GET_NBUFFERS(starpu_task_get_current());

	for (i = 0; i < num; i++)
	{
		int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[i]);
		int nodex = starpu_task_get_current_data_node(i);
		STARPU_ASSERT_MSG(nodex == STARPU_SPECIFIC_NODE_LOCAL || nodex == STARPU_SPECIFIC_NODE_CPU || nodex  == STARPU_SPECIFIC_NODE_SLOW || nodex  == STARPU_SPECIFIC_NODE_LOCAL_OR_CPU || nodex  == STARPU_SPECIFIC_NODE_LOCAL_OR_SIMILAR_OR_CPU || nodex  == STARPU_SPECIFIC_NODE_NONE || (nodex  >= 0 && nodex  < (int) starpu_memory_nodes_get_count()), "The buffers node does not exist");
	}
}

static struct starpu_codelet specific_cl =
{
	.cpu_funcs = {specific_kernel},
	.cuda_funcs = {specific_kernel},
	.opencl_funcs = {specific_kernel},
	.hip_funcs = {specific_kernel},
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
	.specific_nodes = 1,
};

int main(void)
{
	starpu_data_handle_t data_handle[STARPU_NMAXBUFS];
	int data[STARPU_NMAXBUFS];
	starpu_data_handle_t dyn_data_handle[STARPU_NMAXBUFS*2];
	int dyn_data[STARPU_NMAXBUFS*2];

	struct starpu_data_descr *descrs;
	struct starpu_data_descr *dyn_descrs;

	int nodes[STARPU_NMAXBUFS];
	int dyn_nodes[2*STARPU_NMAXBUFS];

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 20;
#else
	unsigned ntasks = 1280;
#endif

	unsigned i;
	int ret;
	int specific_node_count;

	/* Disable prefetching, it makes the test work just by luck */
#ifdef STARPU_HAVE_SETENV
	setenv("STARPU_PREFETCH", "0", 1);
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	specific_node_count = starpu_memory_nodes_get_count() - STARPU_SPECIFIC_NODE_NONE;
	descrs = malloc(STARPU_NMAXBUFS * sizeof(struct starpu_data_descr));
	dyn_descrs = malloc(STARPU_NMAXBUFS * 2 * sizeof(struct starpu_data_descr));

	for (i = 0; i < STARPU_NMAXBUFS; i++)
	{
		data[i] = i;
		starpu_variable_data_register(&(data_handle[i]), STARPU_MAIN_RAM, (uintptr_t) &(data[i]), sizeof(int));
		descrs[i].handle = data_handle[i];
		descrs[i].mode = STARPU_RW;
		/* Try to assign every possible specific node to each handles */
		nodes[i] = STARPU_SPECIFIC_NODE_NONE + (i%specific_node_count);
		if( nodes[i] == STARPU_SPECIFIC_NODE_FAST )
			nodes[i]++;
	}
	for (i = 0; i < STARPU_NMAXBUFS * 2; i++)
	{
		dyn_data[i] = i;
		starpu_variable_data_register(&(dyn_data_handle[i]), STARPU_MAIN_RAM, (uintptr_t) &(dyn_data[i]), sizeof(int));
		dyn_descrs[i].handle = dyn_data_handle[i];
		dyn_descrs[i].mode = STARPU_RW;
		/* Try to assign every possible specific node to each handles */
		dyn_nodes[i] = STARPU_SPECIFIC_NODE_NONE + (i%specific_node_count);
		/* Can't use SPECIFIC_NODE_FAST */
		if( dyn_nodes[i] == STARPU_SPECIFIC_NODE_FAST )
			dyn_nodes[i]++;
	}

	for (i = 0; i < ntasks; i++)
	{
		ret = starpu_task_insert(&specific_cl,
				 STARPU_DATA_MODE_ARRAY, descrs, STARPU_NMAXBUFS,
				 STARPU_NODE_ARRAY, nodes, STARPU_NMAXBUFS,
				 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		ret = starpu_task_insert(&specific_cl,
				 STARPU_DATA_MODE_ARRAY, dyn_descrs, STARPU_NMAXBUFS * 2,
				 STARPU_NODE_ARRAY, dyn_nodes, STARPU_NMAXBUFS * 2,
				 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	goto realend;
enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	ret = STARPU_TEST_SKIPPED;
realend:
	for (i = 0; i < STARPU_NMAXBUFS; i++)
	{
		starpu_data_unregister(data_handle[i]);
	}
	for (i = 0; i < STARPU_NMAXBUFS; i++)
	{
		starpu_data_unregister(dyn_data_handle[i]);
	}
	starpu_shutdown();
	free(descrs);
	free(dyn_descrs);
	return STARPU_TEST_SKIPPED;
}
