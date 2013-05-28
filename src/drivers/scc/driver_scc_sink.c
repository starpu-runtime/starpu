/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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


#include <RCCE.h>

#include <drivers/mp_common/sink_common.h>
#include <drivers/scc/driver_scc_common.h>
#include <drivers/scc/driver_scc_sink.h>



void _starpu_scc_sink_init(struct _starpu_mp_node *node)
{
	node->mp_connection.scc_nodeid = _starpu_scc_common_get_src_node_id();
}

void _starpu_scc_sink_deinit(struct _starpu_mp_node *node)
{
	(void)node;

	_starpu_scc_common_unmap_shared_memory();
	RCCE_finalize();
}

void _starpu_scc_sink_send_to_device(const struct _starpu_mp_node *node, int dst_devid, void *msg, int len)
{
	int ret;
	if ((ret = RCCE_send(msg, len, STARPU_TO_SCC_SINK_ID(dst_devid))) != RCCE_SUCCESS)
		STARPU_MP_COMMON_REPORT_ERROR(node, ret);
}

void _starpu_scc_sink_recv_from_device(const struct _starpu_mp_node *node, int src_devid, void *msg, int len)
{
	int ret;
	if ((ret = RCCE_recv(msg, len, STARPU_TO_SCC_SINK_ID(src_devid))) != RCCE_SUCCESS)
		STARPU_MP_COMMON_REPORT_ERROR(node, ret);
}

/* arg -> [Function pointer on sink, number of interfaces, interfaces
 * (union _starpu_interface), cl_arg]
 *
 * This function change the dev_handle and the ptr of each interfaces
 * given to the sink.
 * dev_handle 	-> 	start of the shared memory (different for each sink)
 * ptr 			-> 	dev_handle + offset
 */
void _starpu_scc_sink_execute(const struct _starpu_mp_node *node, void *arg, int arg_size)
{
	void *local_arg = arg;

	/* point after the kernel */
	local_arg += sizeof(void(*)(void**, void*));

	unsigned nb_interfaces = *(unsigned*)local_arg;
	local_arg += sizeof(nb_interfaces);

	uintptr_t shm_addr = (uintptr_t)_starpu_scc_common_get_shared_memory_addr();

	unsigned i;
	for (i = 0; i < nb_interfaces; ++i)
	{
		/* The first field of an interface is the interface id. */
		switch (*(enum starpu_data_interface_id *)local_arg)
		{
			case STARPU_MATRIX_INTERFACE_ID:
			{
				struct starpu_matrix_interface *matrix = (struct starpu_matrix_interface *)local_arg;
				matrix->dev_handle = shm_addr;
				matrix->ptr = matrix->dev_handle + matrix->offset;
				break;
			}

			case STARPU_BLOCK_INTERFACE_ID:
			{
				struct starpu_block_interface *block = (struct starpu_block_interface *)local_arg;
				block->dev_handle = shm_addr;
				block->ptr = block->dev_handle + block->offset;
				break;
			}

			case STARPU_VECTOR_INTERFACE_ID:
			{
				struct starpu_vector_interface *vector = (struct starpu_vector_interface *)local_arg;
				vector->dev_handle = shm_addr;
				vector->ptr = vector->dev_handle + vector->offset;
				break;
			}

			case STARPU_VARIABLE_INTERFACE_ID:
			{
				struct starpu_variable_interface *variable = (struct starpu_variable_interface *)local_arg;
				variable->dev_handle = shm_addr;
				variable->ptr = variable->dev_handle + variable->offset;
				break;
			}

			case STARPU_CSR_INTERFACE_ID:
			case STARPU_BCSR_INTERFACE_ID:
			case STARPU_MULTIFORMAT_INTERFACE_ID:
			fprintf(stderr, "Data type not supported on SCC.\n");

			default:
				STARPU_ABORT();
		}

		/* point to the next interface */
		local_arg += sizeof(union _starpu_interface);
	}

	_starpu_sink_common_execute(node, arg, arg_size);
}
