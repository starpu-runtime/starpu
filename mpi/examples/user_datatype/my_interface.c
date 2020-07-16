/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi.h>

#include "my_interface.h"

void starpu_my_data_display_codelet_cpu(void *descr[], void *_args)
{
	char c = STARPU_MY_DATA_GET_CHAR(descr[0]);
	int d = STARPU_MY_DATA_GET_INT(descr[0]);
	char msg[100]="";

	if (_args)
		starpu_codelet_unpack_args(_args, &msg);

	fprintf(stderr, "[%s] My value = '%c' %d\n", msg, c, d);
}

void starpu_my_data_compare_codelet_cpu(void *descr[], void *_args)
{
	int *compare;

	starpu_codelet_unpack_args(_args, &compare);

	int d0 = STARPU_MY_DATA_GET_INT(descr[0]);
	char c0 = STARPU_MY_DATA_GET_CHAR(descr[0]);
	int d1 = STARPU_MY_DATA_GET_INT(descr[1]);
	char c1 = STARPU_MY_DATA_GET_CHAR(descr[1]);

	*compare = (d0 == d1 && c0 == c1);
}

void _starpu_my_data_datatype_allocate(MPI_Datatype *mpi_datatype)
{
	int ret;
	int blocklengths[2] = {1, 1};
	MPI_Aint displacements[2];
	MPI_Datatype types[2] = {MPI_INT, MPI_CHAR};
	struct starpu_my_data *myinterface;
	myinterface = malloc(sizeof(struct starpu_my_data));

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

int starpu_my_data_datatype_allocate(starpu_data_handle_t handle, MPI_Datatype *mpi_datatype)
{
	(void)handle;
	_starpu_my_data_datatype_allocate(mpi_datatype);
	return 0;
}

void starpu_my_data_datatype_free(MPI_Datatype *mpi_datatype)
{
	MPI_Type_free(mpi_datatype);
}

int starpu_my_data2_datatype_allocate(starpu_data_handle_t handle, MPI_Datatype *mpi_datatype)
{
	(void)handle;
	(void)mpi_datatype;
	return -1;
}

void starpu_my_data2_datatype_free(MPI_Datatype *mpi_datatype)
{
	STARPU_ASSERT_MSG(0, "should not be called\n");
}

char starpu_my_data_interface_get_char(void *interface)
{
	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) interface;
	struct starpu_my_data *data = (struct starpu_my_data *)my_data->ptr;
	return data->c;
}

int starpu_my_data_interface_get_int(void *interface)
{
	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) interface;
	struct starpu_my_data *data = (struct starpu_my_data *)my_data->ptr;
	return data->d;
}

int starpu_my_data_get_int(starpu_data_handle_t handle)
{
	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	struct starpu_my_data *data = (struct starpu_my_data *)my_data->ptr;
	return data->d;
}

char starpu_my_data_get_char(starpu_data_handle_t handle)
{
	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	struct starpu_my_data *data = (struct starpu_my_data *)my_data->ptr;
	return data->c;
}

static void data_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_my_data_interface *my_data_interface = (struct starpu_my_data_interface *) data_interface;

	unsigned node;
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

static int data_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	(void)handle;
	(void)node;
	(void)ptr;
	(void)count;
	STARPU_ASSERT_MSG(0, "The data interface has been registered with starpu_mpi_datatype_register(). Calling the unpack_data function should not happen\n");
	return 0;
}

static int data_pack_data2(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	*count = sizeof(int) + sizeof(char);
	if (ptr != NULL)
	{
		int d = starpu_my_data_get_int(handle);
		char c = starpu_my_data_get_char(handle);

		*ptr = (void*) starpu_malloc_on_node_flags(node, *count, 0);
		memcpy(*ptr, &d, sizeof(int));
		char *x = *ptr;
		x += sizeof(int);
		memcpy(x, &c, sizeof(char));
	}

	return 0;
}

static int data_unpack_data2(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	(void)count;
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));
	STARPU_ASSERT(count == sizeof(int)+sizeof(char));

	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) starpu_data_get_interface_on_node(handle, node);
	struct starpu_my_data *data = (struct starpu_my_data *)my_data->ptr;
	memcpy(&data->d, ptr, sizeof(int));
	char *x = ptr;
	x += sizeof(int);
	memcpy(&data->c, x, sizeof(char));

	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);
	return 0;
}

static starpu_ssize_t data_describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_my_data_interface *my_data = (struct starpu_my_data_interface *) data_interface;
	struct starpu_my_data *data = (struct starpu_my_data *)my_data->ptr;
	if (data)
		return snprintf(buf, size, "Data%d-%c", data->d, data->c);
	else
		return snprintf(buf, size, "DataUNKNOWN");
}

static void *data_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_my_data_interface *my_data_interface = data_interface;

	return (void*) my_data_interface->ptr;
}

static int copy_any_to_any(void *src_interface, unsigned src_node,
			   void *dst_interface, unsigned dst_node,
			   void *async_data)
{
	struct starpu_my_data *src_data = src_interface;
	struct starpu_my_data *dst_data = dst_interface;
	int ret = 0;

	fprintf(stderr, "copying data src_data.d=%d src_data.c %c\n", src_data->d, src_data->c);

	if (starpu_interface_copy((uintptr_t) src_data->d, 0, src_node,
				  (uintptr_t) dst_data->d, 0, dst_node,
				  sizeof(src_data->d), async_data))
		ret = -EAGAIN;
	if (starpu_interface_copy((uintptr_t) src_data->c, 0, src_node,
				  (uintptr_t) dst_data->c, 0, dst_node,
				  sizeof(src_data->c),
				  async_data))
		ret = -EAGAIN;
	return ret;
}

static const struct starpu_data_copy_methods data_copy_methods =
{
	.any_to_any = copy_any_to_any
};

static struct starpu_data_interface_ops interface_data_ops =
{
	.register_data_handle = data_register_data_handle,
	.allocate_data_on_node = data_allocate_data_on_node,
	.free_data_on_node = data_free_data_on_node,
	.copy_methods = &data_copy_methods,
	.get_size = data_get_size,
	.get_alloc_size = data_get_alloc_size,
	.footprint = data_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_my_data_interface),
	.to_pointer = data_to_pointer,
	.pack_data = data_pack_data,
	.unpack_data = data_unpack_data,
	.describe = data_describe
};

void starpu_my_data_register(starpu_data_handle_t *handleptr, unsigned home_node, struct starpu_my_data *xc)
{
	if (interface_data_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_data_ops.interfaceid = starpu_data_interface_get_next_id();
		starpu_mpi_interface_datatype_register(interface_data_ops.interfaceid, starpu_my_data_datatype_allocate, starpu_my_data_datatype_free);
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

static struct starpu_data_interface_ops interface_data2_ops =
{
	.register_data_handle = data_register_data_handle,
	.allocate_data_on_node = data_allocate_data_on_node,
	.free_data_on_node = data_free_data_on_node,
	.copy_methods = &data_copy_methods,
	.get_size = data_get_size,
	.get_alloc_size = data_get_alloc_size,
	.footprint = data_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_my_data_interface),
	.to_pointer = data_to_pointer,
	.pack_data = data_pack_data2,
	.unpack_data = data_unpack_data2,
	.describe = data_describe
};

void starpu_my_data2_register(starpu_data_handle_t *handleptr, unsigned home_node, struct starpu_my_data *xc)
{
	if (interface_data2_ops.interfaceid == STARPU_UNKNOWN_INTERFACE_ID)
	{
		interface_data2_ops.interfaceid = starpu_data_interface_get_next_id();
		starpu_mpi_interface_datatype_register(interface_data2_ops.interfaceid, starpu_my_data2_datatype_allocate, starpu_my_data2_datatype_free);
	}

	struct starpu_my_data_interface data =
	{
	 	.id = interface_data_ops.interfaceid,
		.ptr = (uintptr_t) xc,
		.dev_handle = (uintptr_t) xc,
		.offset = 0,
	};

	starpu_data_register(handleptr, home_node, &data, &interface_data2_ops);
}

void starpu_my_data2_shutdown(void)
{
	starpu_mpi_interface_datatype_unregister(interface_data2_ops.interfaceid);

}
