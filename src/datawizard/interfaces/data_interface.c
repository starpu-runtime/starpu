/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <datawizard/datawizard.h>

/* 
 * Start monitoring a piece of data
 */

static void _starpu_register_new_data(starpu_data_handle handle,
					uint32_t home_node, uint32_t wb_mask)
{
	STARPU_ASSERT(handle);

	/* initialize the new lock */
	handle->req_list = starpu_data_requester_list_new();
	handle->refcnt = 0;
	_starpu_spin_init(&handle->header_lock);

	/* first take care to properly lock the data */
	_starpu_spin_lock(&handle->header_lock);

	/* there is no hierarchy yet */
	handle->nchildren = 0;
	handle->root_handle = handle;
	handle->father_handle = NULL;
	handle->sibling_index = 0; /* could be anything for the root */
	handle->depth = 1; /* the tree is just a node yet */

	handle->is_not_important = 0;

	handle->sequential_consistency = 1; /* enabled by default */

	PTHREAD_MUTEX_INIT(&handle->sequential_consistency_mutex, NULL);
	handle->last_submitted_mode = STARPU_R;
	handle->last_submitted_writer = NULL;
	handle->last_submitted_readers = NULL;
	handle->post_sync_tasks = NULL;
	handle->post_sync_tasks_cnt = 0;

	handle->wb_mask = wb_mask;

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (node == home_node) {
			/* this is the home node with the only valid copy */
			handle->per_node[node].state = STARPU_OWNER;
			handle->per_node[node].allocated = 1;
			handle->per_node[node].automatically_allocated = 0;
			handle->per_node[node].refcnt = 0;
		}
		else {
			/* the value is not available here yet */
			handle->per_node[node].state = STARPU_INVALID;
			handle->per_node[node].allocated = 0;
			handle->per_node[node].refcnt = 0;
		}
	}

	/* now the data is available ! */
	_starpu_spin_unlock(&handle->header_lock);
}

static starpu_data_handle _starpu_data_handle_allocate(struct starpu_data_interface_ops_t *interface_ops)
{
	starpu_data_handle handle =
		calloc(1, sizeof(struct starpu_data_state_t));

	STARPU_ASSERT(handle);

	handle->ops = interface_ops;

	size_t interfacesize = interface_ops->interface_size;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		handle->interface[node] = calloc(1, interfacesize);
		STARPU_ASSERT(handle->interface[node]);
	}

	return handle;
}

void _starpu_register_data_handle(starpu_data_handle *handleptr, uint32_t home_node,
				void *interface,
				struct starpu_data_interface_ops_t *ops)
{
	starpu_data_handle handle =
		_starpu_data_handle_allocate(ops);

	STARPU_ASSERT(handleptr);
	*handleptr = handle;

	/* fill the interface fields with the appropriate method */
	ops->register_data_handle(handle, home_node, interface);

	_starpu_register_new_data(handle, home_node, 0);
}

/* 
 * Stop monitoring a piece of data
 */

void starpu_data_free_interfaces(starpu_data_handle handle)
{
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
		free(handle->interface[node]);
}

void starpu_data_unregister(starpu_data_handle handle)
{
	unsigned node;

	STARPU_ASSERT(handle);
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		starpu_local_data_state *local = &handle->per_node[node];

		if (local->allocated && local->automatically_allocated){
			/* free the data copy in a lazy fashion */
			_starpu_request_mem_chunk_removal(handle, node);
		}
	}

	starpu_data_requester_list_delete(handle->req_list);

	starpu_data_free_interfaces(handle);

	free(handle);
}

unsigned starpu_get_handle_interface_id(starpu_data_handle handle)
{
	return handle->ops->interfaceid;
}

void *starpu_data_get_interface_on_node(starpu_data_handle handle, unsigned memory_node)
{
	return handle->interface[memory_node];
}
