/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <datawizard/interfaces/data_interface.h>

struct my_interface
{
	enum starpu_data_interface_id id;
	/* Just a integer */
	int x;
};

static struct starpu_data_interface_ops starpu_interface_my_ops;

static void register_my(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	(void) home_node;
	struct my_interface *my_interface = data_interface;
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct my_interface *local_interface = starpu_data_get_interface_on_node(handle, node);
		local_interface->x = my_interface->x;
		local_interface->id = my_interface->id;
	}
}

static size_t my_get_size(starpu_data_handle_t handle)
{
	struct my_interface *my_interface = starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	return my_interface->x;
}

static uint32_t my_footprint(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(my_get_size(handle), 0);
}

static struct starpu_data_interface_ops starpu_interface_my_ops =
{
	.register_data_handle = register_my,
	.allocate_data_on_node = NULL,
	.free_data_on_node = NULL,
	.copy_methods = NULL,
	.get_size = my_get_size,
	.get_max_size = NULL,
	.footprint = my_footprint,
	.compare = NULL,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct my_interface),
	.display = NULL,
	.pack_data = NULL,
	.peek_data = NULL,
	.unpack_data = NULL,
	.describe = NULL,
};

#define N 42
int main(void)
{
	int ret;
	int x;
	starpu_data_handle_t handles[N];
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	for (x = 0; x < N; x++)
	{
		starpu_interface_my_ops.interfaceid = starpu_data_interface_get_next_id();
		struct my_interface my_interface =
		{
			.id = starpu_interface_my_ops.interfaceid,
		};
		starpu_data_register(&handles[x], -1, &my_interface, &starpu_interface_my_ops);
		STARPU_ASSERT(_starpu_data_interface_get_ops(my_interface.id) == &starpu_interface_my_ops);
	}

	for (x = 0; x < N; x++)
		starpu_data_unregister(handles[x]);

	starpu_shutdown();

	return EXIT_SUCCESS;
}
