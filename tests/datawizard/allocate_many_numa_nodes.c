/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2022       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <starpu.h>
#include "../helper.h"

/* Allocate a big buffer not fitting in a single NUMA node, to see what
 * happens, especially if NUMA nodes are correctly reported in traces. */

#if !defined(STARPU_HAVE_UNSETENV) || !defined(STARPU_HAVE_SETENV) || !defined(STARPU_USE_CPU)
#warning unsetenv or setenv are not defined. Or CPU are not enabled. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#elif !defined(STARPU_HAVE_HWLOC) // We need hwloc to know the size of NUMA nodes
#warning hwloc is not used. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif

static void nop(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet cl =
{
	.cpu_funcs = { nop },
	.nbuffers = 1,
	.modes = { STARPU_RW },
};

int main(int argc, char **argv)
{
	int ret;
	starpu_data_handle_t handle;
	int worker;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	unsetenv("STARPU_NCUDA");
	conf.ncpus = -1;
	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, &worker, 1) == 0)
	{
		fprintf(stderr, "Could not find enough workers\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	size_t numa_node_mem;
	hwloc_topology_t topo = starpu_get_hwloc_topology();

	/* This test should work also when NUMA support isn't enabled in
	 * StarPU, so we can't rely on starpu_memory_nodes_get_numa_count(). */
	if (hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE) < 2)
	{
		/* Requires at least 2 NUMA nodes, to avoid overflowing memory
		 * if there is only one NUMA node. */
		fprintf(stderr, "Could not find enough NUMA nodes\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, 0);
	if (!numa_node)
	{
		fprintf(stderr, "Can't find NUMA node 0\n");
		starpu_shutdown();
		return EXIT_FAILURE;
	}
#if HWLOC_API_VERSION >= 0x00020000
	numa_node_mem = numa_node->attr->numanode.local_memory;
#else
	numa_node_mem = numa_node->memory.local_memory;
#endif

	size_t buffer_size = numa_node_mem * 1.5;

	printf("NUMA node 0 has %lu MB, the buffer will use %lu MB\n", numa_node_mem / 1024 / 1024, buffer_size / 1024 / 1024);

	uintptr_t buffer = starpu_malloc_on_node(STARPU_MAIN_RAM, buffer_size);
	if (!buffer)
	{
		fprintf(stderr, "Refuses to allocate that much, too bad\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	memset((void*) buffer, 0, buffer_size);
	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, buffer, buffer_size);

	ret = starpu_task_insert(&cl, STARPU_RW, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

enodev:
	starpu_data_unregister(handle);
	starpu_free_on_node(STARPU_MAIN_RAM, buffer, buffer_size);
	starpu_shutdown();
	return EXIT_SUCCESS;
}
#endif
