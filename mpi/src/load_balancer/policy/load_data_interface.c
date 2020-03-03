/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <common/config.h>

#include "load_data_interface.h"

#if defined(STARPU_USE_MPI_MPI)

int load_data_get_sleep_threshold(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return ld_interface->sleep_task_threshold;
}

int load_data_get_wakeup_threshold(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return ld_interface->wakeup_task_threshold;
}

int load_data_get_current_phase(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return ld_interface->phase;
}

int load_data_get_nsubmitted_tasks(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return ld_interface->nsubmitted_tasks;
}

int load_data_get_nfinished_tasks(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return ld_interface->nfinished_tasks;
}

int load_data_inc_nsubmitted_tasks(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	(ld_interface->nsubmitted_tasks)++;

	return 0;
}

int load_data_inc_nfinished_tasks(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	(ld_interface->nfinished_tasks)++;

	return 0;
}

int load_data_next_phase(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	ld_interface->phase++;

	return 0;
}

int load_data_update_elapsed_time(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	ld_interface->elapsed_time = starpu_timing_now() - ld_interface->start;

	return 0;
}

double load_data_get_elapsed_time(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return ld_interface->elapsed_time;
}

int load_data_update_wakeup_cond(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	int previous_threshold = ld_interface->wakeup_task_threshold;
	ld_interface->wakeup_task_threshold += (ld_interface->nsubmitted_tasks - previous_threshold) * ld_interface->wakeup_ratio;

	return 0;
}

int load_data_wakeup_cond(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return (ld_interface->wakeup_task_threshold > 0) && (ld_interface->nfinished_tasks == ld_interface->wakeup_task_threshold);
}

static void load_data_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	(void) home_node;
	struct load_data_interface *ld_interface = (struct load_data_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct load_data_interface *local_interface = (struct load_data_interface *)
			starpu_data_get_interface_on_node(handle, node);

		local_interface->start = ld_interface->start;
		local_interface->elapsed_time = ld_interface->elapsed_time;
		local_interface->phase = ld_interface->phase;
		local_interface->nsubmitted_tasks = ld_interface->nsubmitted_tasks;
		local_interface->nfinished_tasks = ld_interface->nsubmitted_tasks;
		local_interface->wakeup_task_threshold = ld_interface->wakeup_task_threshold;
		local_interface->wakeup_ratio = ld_interface->wakeup_ratio;
		local_interface->sleep_task_threshold = ld_interface->sleep_task_threshold;
	}
}

static starpu_ssize_t load_data_allocate_data_on_node(void *data_interface, unsigned node)
{
	(void) data_interface;
	(void) node;

	return 0;
}

static void load_data_free_data_on_node(void *data_interface, unsigned node)
{
	(void) data_interface;
	(void) node;
}

static size_t load_data_get_size(starpu_data_handle_t handle)
{
	(void) handle;
	return sizeof(struct load_data_interface);
}

static uint32_t load_data_footprint(starpu_data_handle_t handle)
{
	struct load_data_interface *ld_interface =
		(struct load_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	return starpu_hash_crc32c_be(ld_interface->start,
				     starpu_hash_crc32c_be(ld_interface->elapsed_time,
							   starpu_hash_crc32c_be(ld_interface->nsubmitted_tasks,
										 starpu_hash_crc32c_be(ld_interface->sleep_task_threshold, ld_interface->wakeup_task_threshold))));
}

static int load_data_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct load_data_interface *ld_interface = (struct load_data_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = load_data_get_size(handle);
	if (ptr != NULL)
	{
		char *data;
		starpu_malloc_flags((void**) &data, *count, 0);
		*ptr = data;
		memcpy(data, ld_interface, *count);
	}

	return 0;
}

static int load_data_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	char *data = ptr;
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct load_data_interface *ld_interface = (struct load_data_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == sizeof(struct load_data_interface));
	memcpy(ld_interface, data, count);

	return 0;
}

static int copy_any_to_any(void *src_interface, unsigned src_node,
			   void *dst_interface, unsigned dst_node,
			   void *async_data)
{
	(void) src_interface;
	(void) dst_interface;
	(void) src_node;
	(void) dst_node;
	(void) async_data;

	return 0;
}

static const struct starpu_data_copy_methods load_data_copy_methods =
{
	.any_to_any = copy_any_to_any
};

static struct starpu_data_interface_ops interface_load_data_ops =
{
	.register_data_handle = load_data_register_data_handle,
	.allocate_data_on_node = load_data_allocate_data_on_node,
	.free_data_on_node = load_data_free_data_on_node,
	.copy_methods = &load_data_copy_methods,
	.get_size = load_data_get_size,
	.footprint = load_data_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct load_data_interface),
	.to_pointer = NULL,
	.pack_data = load_data_pack_data,
	.unpack_data = load_data_unpack_data,
	.describe = NULL
};

void load_data_data_register(starpu_data_handle_t *handleptr, unsigned home_node, int sleep_task_threshold, double wakeup_ratio)
{
	struct load_data_interface load_data =
	{
		.start = starpu_timing_now(),
		.elapsed_time = 0,
		.phase = 0,
		.nsubmitted_tasks = 0,
		.nfinished_tasks = 0,
		.sleep_task_threshold = sleep_task_threshold,
		.wakeup_task_threshold = 0,
		.wakeup_ratio = wakeup_ratio
	};

	if (interface_load_data_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_load_data_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &load_data, &interface_load_data_ops);
}

#endif
