/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

/*
 * This is a dumb test for variable size
 * We defined a dumb interface for data whose size increase over kernel execution
 */

#ifdef STARPU_HAVE_MEMCHECK_H
#include <valgrind/memcheck.h>
#else
#define VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(addr, size) (void)0
#endif

#include <core/simgrid.h>

#define FULLSIZE (5*1024*1024ULL)
#define INCREASE 0.80
#ifdef STARPU_QUICK_CHECK
#define N 5
#define LIMIT "60"
#else
#define N 20
#define LIMIT "1000"
#endif

/* Define the interface */

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else

/* Sample Data interface with variable size */
struct variable_size_interface
{
	enum starpu_data_interface_id id;

	/* Just a buffer of a given size */
	uintptr_t ptr;
	size_t size;

	/* Coordinates of the represented object, just for modeling growth */
	unsigned x, y;
};

static struct starpu_data_interface_ops starpu_interface_variable_size_ops;

static void register_variable_size(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct variable_size_interface *variable_size_interface = data_interface;
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct variable_size_interface *local_interface =
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
			local_interface->ptr = variable_size_interface->ptr;
		local_interface->size = variable_size_interface->size;

		local_interface->id = variable_size_interface->id;
		local_interface->x = variable_size_interface->x;
		local_interface->y = variable_size_interface->y;
	}
}

void variable_size_data_register(starpu_data_handle_t *handleptr, unsigned x, unsigned y)
{
	if (starpu_interface_variable_size_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		starpu_interface_variable_size_ops.interfaceid = starpu_data_interface_get_next_id();
	}
	struct variable_size_interface interface =
	{
		.id = starpu_interface_variable_size_ops.interfaceid,
		.x = x,
		.y = y,
	};

	/* Simulate that tiles close to the diagonal are more dense */
	interface.size = FULLSIZE * (starpu_lrand48() % 1024 + 1024) / 2048. * (N-sqrt(abs((int)x-(int)y)*N)) / N;
	/* Round to page size */
	interface.size -= interface.size & (65536-1);

	_starpu_simgrid_data_new(interface.size);

	starpu_data_register(handleptr, -1, &interface, &starpu_interface_variable_size_ops);
}

static size_t variable_size_get_size(starpu_data_handle_t handle)
{
	struct variable_size_interface *interface =
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	return interface->size;
}

static uint32_t variable_size_footprint(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(variable_size_get_size(handle), 0);
}

static int variable_size_compare(void *data_interface_a, void *data_interface_b)
{
	struct variable_size_interface *variable_a = data_interface_a;
	struct variable_size_interface *variable_b = data_interface_b;

	/* Two variables are considered compatible if they have the same size */
	return variable_a->size == variable_b->size;
}

static void display_variable_size(starpu_data_handle_t handle, FILE *f)
{
	struct variable_size_interface *variable_interface =
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%lu\t", (unsigned long) variable_interface->size);
}

static starpu_ssize_t describe_variable_size(void *data_interface, char *buf, size_t size)
{
	struct variable_size_interface *variable_interface = data_interface;
	return snprintf(buf, size, "vv%lu\t", (unsigned long) variable_interface->size);
}

/* returns the size of the allocated area */
static starpu_ssize_t allocate_variable_size_on_node(void *data_interface,
						   unsigned dst_node)
{
	struct variable_size_interface *variable_interface = data_interface;
	variable_interface->ptr = starpu_malloc_on_node_flags(dst_node, variable_interface->size, STARPU_MALLOC_PINNED | STARPU_MALLOC_COUNT | STARPU_MEMORY_OVERFLOW);
	if (dst_node == STARPU_MAIN_RAM)
		_starpu_simgrid_data_alloc(variable_interface->size);
	STARPU_ASSERT(variable_interface->ptr);
	return 0;
}

static void free_variable_size_on_node(void *data_interface,
					unsigned node)
{
	struct variable_size_interface *variable_interface = data_interface;
	starpu_free_on_node(node, variable_interface->ptr, variable_interface->size);
	if (node == STARPU_MAIN_RAM)
		_starpu_simgrid_data_free(variable_interface->size);
}

static int variable_size_copy(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct variable_size_interface *src = src_interface;
	struct variable_size_interface *dst = dst_interface;

	if (src->size != dst->size)
	{
		/* size has been changed by the application in the meantime */
		starpu_free_on_node(dst_node, dst->ptr, dst->size);
		dst->ptr = starpu_malloc_on_node_flags(dst_node, src->size, STARPU_MALLOC_PINNED | STARPU_MALLOC_COUNT | STARPU_MEMORY_OVERFLOW);
		dst->size = src->size;
	}

	return starpu_interface_copy(src->ptr, 0, src_node,
				    dst->ptr, 0, dst_node,
				    src->size, async_data);
}

static const struct starpu_data_copy_methods variable_size_copy_data_methods =
{
	.any_to_any = variable_size_copy,
};

static struct starpu_data_interface_ops starpu_interface_variable_size_ops =
{
	.register_data_handle = register_variable_size,
	.allocate_data_on_node = allocate_variable_size_on_node,
	.free_data_on_node = free_variable_size_on_node,
	.copy_methods = &variable_size_copy_data_methods,
	.get_size = variable_size_get_size,
	.footprint = variable_size_footprint,
	.compare = variable_size_compare,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct variable_size_interface),
	.display = display_variable_size,
	.pack_data = NULL,
	.unpack_data = NULL,
	.describe = describe_variable_size,

	/* We want to observe actual allocations/deallocations */
	.dontcache = 1,
};



static void kernel(void *descr[], void *cl_arg)
{
	struct variable_size_interface *variable_interface = descr[0];
	unsigned workerid = starpu_worker_get_id_check();
	uintptr_t old = variable_interface->ptr;
	unsigned dst_node = starpu_worker_get_memory_node(workerid);

	(void) cl_arg;

	/* Simulate that tiles close to the diagonal fill up faster */
	size_t increase = (FULLSIZE - variable_interface->size) * (starpu_lrand48() % 1024 + 1024) / 2048. * INCREASE;
	/* Round to page size */
	increase -= increase & (65536-1);

	/* Allocation increase */
	variable_interface->ptr = starpu_malloc_on_node_flags(dst_node, variable_interface->size + increase, STARPU_MALLOC_PINNED | STARPU_MALLOC_COUNT | STARPU_MEMORY_OVERFLOW);
	VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE((void*) variable_interface->ptr, variable_interface->size + increase);
	STARPU_ASSERT(variable_interface->ptr);
	/* fprintf(stderr,"increase from %lu by %lu\n", variable_interface->size, increase); */
	starpu_free_on_node_flags(dst_node, old, variable_interface->size, STARPU_MALLOC_PINNED | STARPU_MALLOC_COUNT | STARPU_MEMORY_OVERFLOW);
	variable_interface->size += increase;

	/* These are only simulation bits */
	if (increase)
		_starpu_simgrid_data_increase(increase);
	starpu_sleep(0.010);
}

static double cost_function(struct starpu_task *t, struct starpu_perfmodel_arch *a, unsigned i)
{
	(void)t; (void)a; (void)i;
	return 10000;
}

static struct starpu_perfmodel perf_model =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = cost_function,
};

static struct starpu_codelet cl =
{
	.cpu_funcs = {kernel},

	/* dynamic size doesn't work on MIC */
	/*.cpu_funcs_name = {"kernel"},*/
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &perf_model,
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
};

static void init(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	struct variable_size_interface *variable_interface = descr[0];
	VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE((void*) variable_interface->ptr, variable_interface->size);
}

static struct starpu_codelet cl_init =
{
	.cpu_funcs = {init},

	/* dynamic size doesn't work on MIC */
	/*.cpu_funcs_name = {"kernel"},*/
	.nbuffers = 1,
	.modes = {STARPU_W},
	.model = &starpu_perfmodel_nop,
};

int main(void)
{
	int ret;
	int i;
	int x, y;
	starpu_data_handle_t handles[N][N];
	char s[128];

	snprintf(s, sizeof(s), "/tmp/%s-variable_size", getenv("USER"));

	setenv("STARPU_CALIBRATE_MINIMUM", "1", 1);
	setenv("STARPU_LIMIT_CPU_MEM", LIMIT, 1);
	setenv("STARPU_DISK_SWAP", s, 0);
	setenv("STARPU_DISK_SWAP_SIZE", "100000", 1);
#if 0 //def STARPU_LINUX_SYS
	setenv("STARPU_DISK_SWAP_BACKEND", "unistd_o_direct", 0);
#else
	setenv("STARPU_DISK_SWAP_BACKEND", "unistd", 0);
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	for (x = 0; x < N; x++)
		for (y = 0; y < N; y++)
		{
			variable_size_data_register(&handles[x][y], x, y);

			ret = starpu_task_insert(&cl_init, STARPU_W, handles[x][y], 0);
			if (ret == ENODEV)
				goto enodev;
#ifdef STARPU_SIMGRID
			starpu_sleep(0.0005);
#endif
		}

	starpu_task_wait_for_all();

	/* Cholesky-like accesses */
	for (i = 0; i < N; i++)
		for (x = i; x < N; x++)
			for (y = x; y < N; y++)
				starpu_task_insert(&cl, STARPU_RW, handles[x][y], STARPU_PRIORITY, (2*N-x-y), 0);

	starpu_task_wait_for_all();

#if 0
	/* Look at the values */
	for (x = 0; x < N; x++)
		for (y = 0; y < N; y++)
		{
			starpu_data_acquire(handles[x][y], STARPU_R);
			starpu_data_release(handles[x][y]);
		}
#endif

	for (x = 0; x < N; x++)
		for (y = 0; y < N; y++)
			starpu_data_unregister(handles[x][y]);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	for (x = 0; x < N; x++)
		for (y = 0; y < N; y++)
			starpu_data_unregister(handles[x][y]);

	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
#endif
