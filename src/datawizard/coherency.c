/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <common/config.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/write_back.h>
#include <core/dependencies/data-concurrency.h>

uint32_t starpu_select_node_to_handle_request(uint32_t src_node, uint32_t dst_node) 
{
	/* in case one of the node is a GPU, it needs to perform the transfer,
	 * if both of them are GPU, it's a bit more complicated (TODO !) */

	unsigned src_is_a_gpu = (get_node_kind(src_node) == CUDA_RAM);
	unsigned dst_is_a_gpu = (get_node_kind(dst_node) == CUDA_RAM);

	/* we do not handle GPU->GPU transfers yet ! */
	STARPU_ASSERT( !(src_is_a_gpu && dst_is_a_gpu) );

	if (src_is_a_gpu)
		return src_node;

	if (dst_is_a_gpu)
		return dst_node;

	/* otherwise perform it locally, since we should be on a "sane" arch
	 * where anyone can do the transfer. NB: in StarPU this should actually never
	 * happen */
	return get_local_memory_node();
}

uint32_t starpu_select_src_node(starpu_data_handle handle)
{
	unsigned src_node = 0;
	unsigned i;

	unsigned nnodes = get_memory_nodes_count();

	/* first find a valid copy, either a STARPU_OWNER or a STARPU_SHARED */
	uint32_t node;

	uint32_t src_node_mask = 0;
	for (node = 0; node < nnodes; node++)
	{
		if (handle->per_node[node].state != STARPU_INVALID) {
			/* we found a copy ! */
			src_node_mask |= (1<<node);
		}
	}

	/* we should have found at least one copy ! */
	STARPU_ASSERT(src_node_mask != 0);

	/* find the node that will be the actual source */
	for (i = 0; i < nnodes; i++)
	{
		if (src_node_mask & (1<<i))
		{
			/* this is a potential candidate */
			src_node = i;

			/* however GPU are expensive sources, really !
			 * 	other should be ok */
			if (get_node_kind(i) != CUDA_RAM)
				break;

			/* XXX do a better algorithm to distribute the memory copies */
			/* TODO : use the "requesting_node" as an argument to do so */
		}
	}

	return src_node;
}

/* this may be called once the data is fetched with header and STARPU_RW-lock hold */
void starpu_update_data_state(starpu_data_handle handle, uint32_t requesting_node, uint8_t write)
{
	unsigned nnodes = get_memory_nodes_count();

	/* the data is present now */
	handle->per_node[requesting_node].requested = 0;

	if (write) {
		/* the requesting node now has the only valid copy */
		uint32_t node;
		for (node = 0; node < nnodes; node++)
			handle->per_node[node].state = STARPU_INVALID;

		handle->per_node[requesting_node].state = STARPU_OWNER;
	}
	else { /* read only */
		if (handle->per_node[requesting_node].state != STARPU_OWNER)
		{
			/* there was at least another copy of the data */
			uint32_t node;
			for (node = 0; node < nnodes; node++)
			{
				if (handle->per_node[node].state != STARPU_INVALID)
					handle->per_node[node].state = STARPU_SHARED;
			}
			handle->per_node[requesting_node].state = STARPU_SHARED;
		}
	}
}


/*
 * This function is called when the data is needed on the local node, this
 * returns a pointer to the local copy 
 *
 *			R 	STARPU_W 	STARPU_RW
 *	Owner		OK	OK	OK
 *	Shared		OK	1	1
 *	Invalid		2	3	4
 *
 * case 1 : shared + (read)write : 
 * 	no data copy but shared->Invalid/Owner
 * case 2 : invalid + read : 
 * 	data copy + invalid->shared + owner->shared (STARPU_ASSERT(there is a valid))
 * case 3 : invalid + write : 
 * 	no data copy + invalid->owner + (owner,shared)->invalid
 * case 4 : invalid + R/STARPU_W : 
 * 	data copy + if (STARPU_W) (invalid->owner + owner->invalid) 
 * 		    else (invalid,owner->shared)
 */

int starpu_fetch_data_on_node(starpu_data_handle handle, uint32_t requesting_node,
			uint8_t read, uint8_t write, unsigned is_prefetch)
{
	uint32_t local_node = get_local_memory_node();

	while (starpu_spin_trylock(&handle->header_lock))
		_starpu_datawizard_progress(local_node, 1);

	if (!is_prefetch)
		handle->per_node[requesting_node].refcnt++;

	if (handle->per_node[requesting_node].state != STARPU_INVALID)
	{
		/* the data is already available so we can stop */
		starpu_update_data_state(handle, requesting_node, write);
		msi_cache_hit(requesting_node);
		starpu_spin_unlock(&handle->header_lock);
		return 0;
	}

	/* the only remaining situation is that the local copy was invalid */
	STARPU_ASSERT(handle->per_node[requesting_node].state == STARPU_INVALID);

	msi_cache_miss(requesting_node);

	starpu_data_request_t r;

	/* is there already a pending request ? */
	r = starpu_search_existing_data_request(handle, requesting_node, read, write);
	/* at the exit of starpu_search_existing_data_request the lock is taken is the request existed ! */

	if (!r) {
		//fprintf(stderr, "no request matched that one so we post a request %s\n", is_prefetch?"STARPU_PREFETCH":"");
		/* find someone who already has the data */
		uint32_t src_node = 0;

		/* if the data is in read only mode, there is no need for a source */
		if (read)
		{
			src_node = starpu_select_src_node(handle);
			STARPU_ASSERT(src_node != requesting_node);
		}
	
		unsigned src_is_a_gpu = (get_node_kind(src_node) == CUDA_RAM);
		unsigned dst_is_a_gpu = (get_node_kind(requesting_node) == CUDA_RAM);

		/* we have to perform 2 successive requests for GPU->GPU transfers */
		if (read && (src_is_a_gpu && dst_is_a_gpu)) {
			unsigned reuse_r_src_to_ram;
			starpu_data_request_t r_src_to_ram;
			starpu_data_request_t r_ram_to_dst;

			/* XXX we hardcore 0 as the RAM node ... */
			r_ram_to_dst = starpu_create_data_request(handle, 0, requesting_node, requesting_node, read, write, is_prefetch);

			if (!is_prefetch)
				r_ram_to_dst->refcnt++;

			r_src_to_ram = starpu_search_existing_data_request(handle, 0, read, write);
			if (!r_src_to_ram)
			{
				reuse_r_src_to_ram = 0;
				r_src_to_ram = starpu_create_data_request(handle, src_node, 0, src_node, read, write, is_prefetch);
			}
			else {
				reuse_r_src_to_ram = 1;
			}

			/* we chain both requests */
			r_src_to_ram->next_req[r_src_to_ram->next_req_count++]= r_ram_to_dst;

			if (reuse_r_src_to_ram)
				starpu_spin_unlock(&r_src_to_ram->lock);

			starpu_spin_unlock(&handle->header_lock);

			/* we only submit the first request, the remaining will be automatically submitted afterward */
			if (!reuse_r_src_to_ram)
				starpu_post_data_request(r_src_to_ram, src_node);

			/* the application only waits for the termination of the last request */
			r = r_ram_to_dst;
		}
		else {
			/* who will perform that request ? */
			uint32_t handling_node =
				starpu_select_node_to_handle_request(src_node, requesting_node);

			r = starpu_create_data_request(handle, src_node, requesting_node, handling_node, read, write, is_prefetch);

			if (!is_prefetch)
				r->refcnt++;

			starpu_spin_unlock(&handle->header_lock);

			starpu_post_data_request(r, handling_node);
		}
	}
	else {
		/* the lock was taken by starpu_search_existing_data_request */

		/* there is already a similar request */
		if (is_prefetch)
		{
			starpu_spin_unlock(&r->lock);

			starpu_spin_unlock(&handle->header_lock);
			return 0;
		}

		r->refcnt++;

		//starpu_spin_lock(&r->lock);
		if (r->is_a_prefetch_request)
		{
			/* transform that prefetch request into a "normal" request */
			r->is_a_prefetch_request = 0;

			/* transform that request into the proper access mode (prefetch could be read only) */
			r->read = read;
			r->write = write;
		}

		//fprintf(stderr, "found a similar request : refcnt (req) %d\n", r->refcnt);
		starpu_spin_unlock(&r->lock);
		starpu_spin_unlock(&handle->header_lock);
	}

	return (is_prefetch?0:starpu_wait_data_request_completion(r, 1));
}

static int prefetch_data_on_node(starpu_data_handle handle, uint8_t read, uint8_t write, uint32_t node)
{
	return starpu_fetch_data_on_node(handle, node, read, write, 1);
}

static int fetch_data(starpu_data_handle handle, starpu_access_mode mode)
{
	uint32_t requesting_node = get_local_memory_node(); 

	uint8_t read, write;
	read = (mode != STARPU_W); /* then R or STARPU_RW */
	write = (mode != STARPU_R); /* then STARPU_W or STARPU_RW */

	return starpu_fetch_data_on_node(handle, requesting_node, read, write, 0);
}

inline uint32_t starpu_get_data_refcnt(starpu_data_handle handle, uint32_t node)
{
	return handle->per_node[node].refcnt;
}

/* in case the data was accessed on a write mode, do not forget to 
 * make it accessible again once it is possible ! */
void starpu_release_data_on_node(starpu_data_handle handle, uint32_t default_wb_mask, uint32_t memory_node)
{
	uint32_t wb_mask;

	/* normally, the requesting node should have the data in an exclusive manner */
	STARPU_ASSERT(handle->per_node[memory_node].state != STARPU_INVALID);

	wb_mask = default_wb_mask | handle->wb_mask;

	/* are we doing write-through or just some normal write-back ? */
	if (wb_mask & ~(1<<memory_node)) {
		write_through_data(handle, memory_node, wb_mask);
	}

	uint32_t local_node = get_local_memory_node();
	while (starpu_spin_trylock(&handle->header_lock))
		_starpu_datawizard_progress(local_node, 1);

	handle->per_node[memory_node].refcnt--;

	notify_data_dependencies(handle);

	starpu_spin_unlock(&handle->header_lock);
}

int starpu_prefetch_task_input_on_node(struct starpu_task *task, uint32_t node)
{
	starpu_buffer_descr *descrs = task->buffers;
	unsigned nbuffers = task->cl->nbuffers;

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_buffer_descr *descr;
		starpu_data_handle handle;

		descr = &descrs[index];
		handle = descr->handle;
		
		uint32_t mode = task->buffers[index].mode;
	
		uint8_t read = (mode != STARPU_W);
		uint8_t write = (mode != STARPU_R);

		prefetch_data_on_node(handle, read, write, node);
	}

	return 0;
}



int _starpu_fetch_task_input(struct starpu_task *task, uint32_t mask)
{
	TRACE_START_FETCH_INPUT(NULL);

//	fprintf(stderr, "_starpu_fetch_task_input\n");

	uint32_t local_memory_node = get_local_memory_node();

	starpu_buffer_descr *descrs = task->buffers;
	unsigned nbuffers = task->cl->nbuffers;

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		int ret;
		starpu_buffer_descr *descr;
		starpu_data_handle handle;

		descr = &descrs[index];

		handle = descr->handle;
	
		ret = fetch_data(handle, descr->mode);
		if (STARPU_UNLIKELY(ret))
			goto enomem;

		void *src_interface = starpu_data_get_interface_on_node(handle, local_memory_node);
		task->interface[index] = src_interface;
	}

	TRACE_END_FETCH_INPUT(NULL);

	return 0;

enomem:
	/* try to unreference all the input that were successfully taken */
	/* XXX broken ... */
	fprintf(stderr, "something went wrong with buffer %u\n", index);
	//push_codelet_output(task, index, mask);
	starpu_push_task_output(task, mask);
	return -1;
}

void starpu_push_task_output(struct starpu_task *task, uint32_t mask)
{
	TRACE_START_PUSH_OUTPUT(NULL);

	//fprintf(stderr, "starpu_push_task_output\n");

        starpu_buffer_descr *descrs = task->buffers;
        unsigned nbuffers = task->cl->nbuffers;

	uint32_t local_node = get_local_memory_node();

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_release_data_on_node(descrs[index].handle, mask, local_node);
	}

	TRACE_END_PUSH_OUTPUT(NULL);
}

/* NB : this value can only be an indication of the status of a data
	at some point, but there is no strong garantee ! */
unsigned starpu_is_data_present_or_requested(starpu_data_handle handle, uint32_t node)
{
	unsigned ret = 0;

// XXX : this is just a hint, so we don't take the lock ...
//	pthread_spin_lock(&handle->header_lock);

	if (handle->per_node[node].state != STARPU_INVALID 
		|| handle->per_node[node].requested || handle->per_node[node].request)
		ret = 1;

//	pthread_spin_unlock(&handle->header_lock);

	return ret;
}

inline void starpu_set_data_requested_flag_if_needed(starpu_data_handle handle, uint32_t node)
{
// XXX : this is just a hint, so we don't take the lock ...
//	pthread_spin_lock(&handle->header_lock);

	if (handle->per_node[node].state == STARPU_INVALID) 
		handle->per_node[node].requested = 1;

//	pthread_spin_unlock(&handle->header_lock);
}

unsigned starpu_test_if_data_is_allocated_on_node(starpu_data_handle handle, uint32_t memory_node)
{
	return handle->per_node[memory_node].allocated;
} 
