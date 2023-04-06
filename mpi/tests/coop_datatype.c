/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi.h>
#include "helper.h"

struct starpu_my_data_interface
{
	enum starpu_data_interface_id id; /**< Identifier of the interface */

	uintptr_t ptr;                    /**< local pointer of the data */
	uintptr_t dev_handle;             /**< device handle of the data. */
	size_t offset;                    /**< offset in the data */
};

struct starpu_my_data
{
	int d;
	char c;
};

void _starpu_my_data_datatype_allocate(unsigned node, MPI_Datatype *mpi_datatype)
{
	int ret;
	int blocklengths[2] = {1, 1};
	MPI_Aint displacements[2];
	MPI_Datatype types[2] = {MPI_INT, MPI_CHAR};
	struct starpu_my_data *myinterface;
	myinterface = calloc(1, sizeof(struct starpu_my_data));

	MPI_Get_address(myinterface, displacements);
	MPI_Get_address(&myinterface[0].c, displacements+1);
	displacements[1] -= displacements[0];
	displacements[0] = 0;

	ret = MPI_Type_create_struct(2, blocklengths, displacements, types, mpi_datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(mpi_datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	free(myinterface);
}

int starpu_my_data_datatype_allocate(starpu_data_handle_t handle, unsigned node, MPI_Datatype *mpi_datatype)
{
	(void)handle;
	_starpu_my_data_datatype_allocate(node, mpi_datatype);
	return 0;
}

void starpu_my_data_datatype_free(MPI_Datatype *mpi_datatype)
{
	int ret;
	ret = MPI_Type_free(mpi_datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_free failed");
}

static void data_register_data_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpu_my_data_interface *my_data_interface = (struct starpu_my_data_interface *) data_interface;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_my_data_interface *local_interface =
			(struct starpu_my_data_interface *) starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = my_data_interface->ptr;
			local_interface->dev_handle = my_data_interface->dev_handle;
			local_interface->offset = my_data_interface->offset;
		}
		else
		{
			local_interface->ptr = 0;
			local_interface->dev_handle = 0;
			local_interface->offset = 0;
		}
	}
}

static starpu_ssize_t data_allocate_data_on_node(void *data_interface, unsigned node)
{
	uintptr_t addr = 0, handle;

	struct starpu_my_data_interface *my_data_interface = (struct starpu_my_data_interface *) data_interface;

	starpu_ssize_t allocated_memory = sizeof(int)+sizeof(char);
	handle = starpu_malloc_on_node(node, allocated_memory);
	if (!handle)
		return -ENOMEM;

	if (starpu_node_get_kind(node) != STARPU_OPENCL_RAM)
		addr = handle;

	/* update the data properly in consequence */
	my_data_interface->ptr = addr;
	my_data_interface->dev_handle = handle;
	my_data_interface->offset = 0;

	return allocated_memory;
}

static void data_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_my_data_interface *my_data_interface = (struct starpu_my_data_interface *) data_interface;
	starpu_free_on_node(node, my_data_interface->dev_handle, sizeof(int)+sizeof(char));
	my_data_interface->ptr = 0;
	my_data_interface->dev_handle = 0;
}

static size_t data_get_size(starpu_data_handle_t handle)
{
	(void)handle;
	return sizeof(int) + sizeof(char);
}

static size_t data_get_alloc_size(starpu_data_handle_t handle)
{
	(void)handle;
	return sizeof(int) + sizeof(char);
}

static uint32_t data_footprint(starpu_data_handle_t handle)
{
	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	return starpu_hash_crc32c_be(my_data->ptr, 0);
}

static int data_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	(void)handle;
	(void)node;
	(void)ptr;
	(void)count;
	STARPU_ASSERT_MSG(0, "The data interface has been registered with starpu_mpi_datatype_register(). Calling the pack_data function should not happen\n");
	return 0;
}

static int data_peek_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	(void)handle;
	(void)node;
	(void)ptr;
	STARPU_ASSERT_MSG(0, "The data interface has been registered with starpu_mpi_datatype_register(). Calling the unpack_data function should not happen\n");
	return 0;
}

static int data_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	(void)handle;
	(void)node;
	(void)ptr;
	(void)count;
	STARPU_ASSERT_MSG(0, "The data interface has been registered with starpu_mpi_datatype_register(). Calling the unpack_data function should not happen\n");
	return 0;
}

static starpu_ssize_t data_describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) data_interface;
	struct starpu_my_data *data = (struct starpu_my_data *)my_data->ptr;
	return snprintf(buf, size, "Data%d-%c", data->d, data->c);
}

static void *data_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_my_data_interface *my_data_interface = data_interface;

	return (void*) my_data_interface->ptr;
}

static struct starpu_data_interface_ops interface_data_ops =
{
	.register_data_handle = data_register_data_handle,
	.allocate_data_on_node = data_allocate_data_on_node,
	.free_data_on_node = data_free_data_on_node,
	.get_size = data_get_size,
	.get_alloc_size = data_get_alloc_size,
	.footprint = data_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_my_data_interface),
	.to_pointer = data_to_pointer,
	.pack_data = data_pack_data,
	.peek_data = data_peek_data,
	.unpack_data = data_unpack_data,
	.describe = data_describe
};

void starpu_my_data_register(starpu_data_handle_t *handleptr, unsigned home_node, struct starpu_my_data *xc)
{
	if (interface_data_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_data_ops.interfaceid = starpu_data_interface_get_next_id();
		starpu_mpi_interface_datatype_node_register(interface_data_ops.interfaceid, starpu_my_data_datatype_allocate, starpu_my_data_datatype_free);
	}

	struct starpu_my_data_interface data =
	{
		.id = interface_data_ops.interfaceid,
		.ptr = (uintptr_t) xc,
		.dev_handle = (uintptr_t) xc,
		.offset = 0,
	};

	starpu_data_register(handleptr, home_node, &data, &interface_data_ops);
}

void starpu_my_data_shutdown(void)
{
	starpu_mpi_interface_datatype_unregister(interface_data_ops.interfaceid);

}

int main(int argc, char **argv)
{
	int rank, nodes, mpi_init;
	int ret;
	const int tag = 12;
	int i = 0;
	struct starpu_conf conf;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &nodes);

	if (nodes < 2 || (starpu_cpu_worker_get_count() == 0))
	{
		if (rank == 0)
		{
			if (nodes < 2)
				fprintf(stderr, "We need at least 2 processes.\n");
			else
				fprintf(stderr, "We need at least 1 CPU.\n");
		}
		starpu_mpi_shutdown();
		return 77;
	}

	struct starpu_my_data my0;
	starpu_data_handle_t handle0;

	starpu_my_data_register(&handle0, STARPU_MAIN_RAM, &my0);

	if (rank == 0)
	{
		my0.d = 43;
		my0.c = 'm';

		starpu_mpi_coop_sends_data_handle_nb_sends(handle0, nodes-1);
		for (i = 1; i < nodes; i++)
		{
			ret = starpu_mpi_isend_detached(handle0, i, tag, MPI_COMM_WORLD, NULL, NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
		}
	}
	else
	{
		my0.d = 23;
		my0.c = 'd';

		ret = starpu_mpi_recv(handle0, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(handle0, STARPU_R);

		printf("[%d] received: %d %c\n", rank, my0.d, my0.c);

		assert(my0.d == 43);
		assert(my0.c == 'm');

		starpu_data_release(handle0);
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	starpu_data_unregister(handle0);
	starpu_my_data_shutdown();

	starpu_mpi_shutdown();
 	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
