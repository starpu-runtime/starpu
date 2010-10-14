/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
#include <datawizard/copy_driver.h>
#include <datawizard/write_back.h>
#include <core/dependencies/data_concurrency.h>

uint32_t _starpu_select_node_to_handle_request(uint32_t src_node, uint32_t dst_node) 
{
	/* in case one of the node is a GPU, it needs to perform the transfer,
	 * if both of them are GPU, it's a bit more complicated */

	unsigned src_is_a_gpu = (_starpu_get_node_kind(src_node) == STARPU_CUDA_RAM || _starpu_get_node_kind(src_node) == STARPU_OPENCL_RAM);
	unsigned dst_is_a_gpu = (_starpu_get_node_kind(dst_node) == STARPU_CUDA_RAM || _starpu_get_node_kind(dst_node) == STARPU_OPENCL_RAM);

	/* we do not handle GPU->GPU transfers yet ! */
	STARPU_ASSERT( !(src_is_a_gpu && dst_is_a_gpu) );

	if (src_is_a_gpu)
		return src_node;

	if (dst_is_a_gpu)
		return dst_node;

	/* otherwise perform it locally, since we should be on a "sane" arch
	 * where anyone can do the transfer. NB: in StarPU this should actually never
	 * happen */
	return _starpu_get_local_memory_node();
}

uint32_t _starpu_select_src_node(starpu_data_handle handle)
{
	unsigned src_node = 0;
	unsigned i;

	unsigned nnodes = _starpu_get_memory_nodes_count();

	/* first find a valid copy, either a STARPU_OWNER or a STARPU_SHARED */
	uint32_t node;

	uint32_t src_node_mask = 0;
	for (node = 0; node < nnodes; node++)
	{
		if (handle->per_node[node]->state != STARPU_INVALID) {
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
			if (_starpu_get_node_kind(i) != STARPU_CUDA_RAM)
				break;
			if (_starpu_get_node_kind(i) != STARPU_OPENCL_RAM)
				break;

			/* XXX do a better algorithm to distribute the memory copies */
			/* TODO : use the "requesting_node" as an argument to do so */
		}
	}

	return src_node;
}

/* this may be called once the data is fetched with header and STARPU_RW-lock hold */
void _starpu_update_data_state(starpu_data_handle handle,
				struct starpu_data_replicate_s *requesting_replicate,
				starpu_access_mode mode)
{
	unsigned nnodes = _starpu_get_memory_nodes_count();

	/* the data is present now */
	requesting_replicate->requested = 0;

	if (mode & STARPU_W) {
		/* the requesting node now has the only valid copy */
		uint32_t node;
		for (node = 0; node < nnodes; node++)
			handle->per_node[node]->state = STARPU_INVALID;

		requesting_replicate->state = STARPU_OWNER;
	}
	else { /* read only */
		if (requesting_replicate->state != STARPU_OWNER)
		{
			/* there was at least another copy of the data */
			uint32_t node;
			for (node = 0; node < nnodes; node++)
			{
				struct starpu_data_replicate_s *replicate = handle->per_node[node];
				if (replicate->state != STARPU_INVALID)
					replicate->state = STARPU_SHARED;
			}
			requesting_replicate->state = STARPU_SHARED;
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

int _starpu_fetch_data_on_node(starpu_data_handle handle, struct starpu_data_replicate_s *dst_replicate,
				starpu_access_mode mode, unsigned is_prefetch,
				void (*callback_func)(void *), void *callback_arg)
{
	uint32_t local_node = _starpu_get_local_memory_node();
        _STARPU_LOG_IN();

	unsigned requesting_node = dst_replicate->memory_node;

	while (_starpu_spin_trylock(&handle->header_lock))
		_starpu_datawizard_progress(local_node, 1);

	if (!is_prefetch)
		dst_replicate->refcnt++;

	if (dst_replicate->state != STARPU_INVALID)
	{
		/* the data is already available so we can stop */
		_starpu_update_data_state(handle, dst_replicate, mode);
		_starpu_msi_cache_hit(requesting_node);
		_starpu_spin_unlock(&handle->header_lock);

		if (callback_func)
			callback_func(callback_arg);

                _STARPU_LOG_OUT_TAG("data available");
		return 0;
	}

	/* the only remaining situation is that the local copy was invalid */
	STARPU_ASSERT(dst_replicate->state == STARPU_INVALID);

	_starpu_msi_cache_miss(requesting_node);

	starpu_data_request_t r;

	/* is there already a pending request ? */
	r = _starpu_search_existing_data_request(dst_replicate, mode);
	/* at the exit of _starpu_search_existing_data_request the lock is taken is the request existed ! */

	if (!r) {
		/* find someone who already has the data */
		uint32_t src_node = 0;

		/* if the data is in write only mode, there is no need for a source */
		if (mode & STARPU_R)
		{
			src_node = _starpu_select_src_node(handle);
			STARPU_ASSERT(src_node != requesting_node);
		}
	
		unsigned src_is_a_gpu = (_starpu_get_node_kind(src_node) == STARPU_CUDA_RAM || _starpu_get_node_kind(src_node) == STARPU_OPENCL_RAM);
		unsigned dst_is_a_gpu = (_starpu_get_node_kind(requesting_node) == STARPU_CUDA_RAM || _starpu_get_node_kind(requesting_node) == STARPU_OPENCL_RAM);

		struct starpu_data_replicate_s *src_replicate = handle->per_node[src_node];

		/* we have to perform 2 successive requests for GPU->GPU transfers */
		if ((mode & STARPU_R) && (src_is_a_gpu && dst_is_a_gpu)) {
			unsigned reuse_r_src_to_ram;
			starpu_data_request_t r_src_to_ram;
			starpu_data_request_t r_ram_to_dst;

			struct starpu_data_replicate_s *ram_replicate = handle->per_node[0];

			/* XXX we hardcore 0 as the RAM node ... */
			r_ram_to_dst = _starpu_create_data_request(handle, ram_replicate,
						dst_replicate, requesting_node, mode, is_prefetch);

			if (!is_prefetch)
				r_ram_to_dst->refcnt++;

			r_src_to_ram = _starpu_search_existing_data_request(ram_replicate, mode);

			reuse_r_src_to_ram = r_src_to_ram?1:0;

			if (!r_src_to_ram)
			{
				r_src_to_ram = _starpu_create_data_request(handle, src_replicate,
							ram_replicate, src_node, mode, is_prefetch);
			}

			/* we chain both requests */
			r_src_to_ram->next_req[r_src_to_ram->next_req_count++]= r_ram_to_dst;

			_starpu_data_request_append_callback(r_ram_to_dst, callback_func, callback_arg);

			if (reuse_r_src_to_ram)
				_starpu_spin_unlock(&r_src_to_ram->lock);

			_starpu_spin_unlock(&handle->header_lock);

			/* we only submit the first request, the remaining will be automatically submitted afterward */
			if (!reuse_r_src_to_ram)
				_starpu_post_data_request(r_src_to_ram, src_node);

			/* the application only waits for the termination of the last request */
			r = r_ram_to_dst;
		}
		else {
			/* who will perform that request ? */
			uint32_t handling_node =
				_starpu_select_node_to_handle_request(src_node, requesting_node);

			r = _starpu_create_data_request(handle, src_replicate,
					dst_replicate, handling_node, mode, is_prefetch);

			_starpu_data_request_append_callback(r, callback_func, callback_arg);

			if (!is_prefetch)
				r->refcnt++;

			_starpu_spin_unlock(&handle->header_lock);

			_starpu_post_data_request(r, handling_node);
		}
	}
	else {
		/* the lock was taken by _starpu_search_existing_data_request */
		_starpu_data_request_append_callback(r, callback_func, callback_arg);

		/* there is already a similar request */
		if (is_prefetch)
		{
			_starpu_spin_unlock(&r->lock);

			_starpu_spin_unlock(&handle->header_lock);

                        _STARPU_LOG_OUT_TAG("similar request");
                        return 0;
		}

		r->refcnt++;

		//_starpu_spin_lock(&r->lock);
		if (r->is_a_prefetch_request)
		{
			/* transform that prefetch request into a "normal" request */
			r->is_a_prefetch_request = 0;

			/* transform that request into the proper access mode (prefetch could be read only) */
#warning check that
			r->mode |= mode;
		}

		//_STARPU_DEBUG("found a similar request : refcnt (req) %d\n", r->refcnt);
		_starpu_spin_unlock(&r->lock);
		_starpu_spin_unlock(&handle->header_lock);
	}

	int ret = is_prefetch?0:_starpu_wait_data_request_completion(r, 1);
        _STARPU_LOG_OUT();
        return ret;
}

static int prefetch_data_on_node(starpu_data_handle handle, struct starpu_data_replicate_s *replicate, starpu_access_mode mode)
{
	return _starpu_fetch_data_on_node(handle, replicate, mode, 1, NULL, NULL);
}

static int fetch_data(starpu_data_handle handle, struct starpu_data_replicate_s *replicate, starpu_access_mode mode)
{
	STARPU_ASSERT(!(mode & STARPU_SCRATCH));

	return _starpu_fetch_data_on_node(handle, replicate, mode, 0, NULL, NULL);
}

inline uint32_t _starpu_get_data_refcnt(starpu_data_handle handle, uint32_t node)
{
	return handle->per_node[node]->refcnt;
}

size_t _starpu_data_get_size(starpu_data_handle handle)
{
	return handle->data_size;
}

uint32_t _starpu_data_get_footprint(starpu_data_handle handle)
{
	return handle->footprint;
}

/* in case the data was accessed on a write mode, do not forget to 
 * make it accessible again once it is possible ! */
void _starpu_release_data_on_node(starpu_data_handle handle, uint32_t default_wt_mask, struct starpu_data_replicate_s *replicate)
{
	uint32_t wt_mask;
	wt_mask = default_wt_mask | handle->wt_mask;

	/* Note that it is possible that there is no valid copy of the data (if
	 * starpu_data_invalidate was called for instance). In that case, we do
	 * not enforce any write-through mechanism. */

	unsigned memory_node = replicate->memory_node;

	if (replicate->state != STARPU_INVALID)
	if ((wt_mask & ~(1<<memory_node)))
		_starpu_write_through_data(handle, memory_node, wt_mask);

	uint32_t local_node = _starpu_get_local_memory_node();
	while (_starpu_spin_trylock(&handle->header_lock))
		_starpu_datawizard_progress(local_node, 1);

	replicate->refcnt--;

	STARPU_ASSERT(replicate->refcnt >= 0);

	_starpu_notify_data_dependencies(handle);

	_starpu_spin_unlock(&handle->header_lock);
}

static void _starpu_set_data_requested_flag_if_needed(struct starpu_data_replicate_s *replicate)
{
// XXX : this is just a hint, so we don't take the lock ...
//	pthread_spin_lock(&handle->header_lock);

	if (replicate->state == STARPU_INVALID) 
		replicate->requested = 1;

//	pthread_spin_unlock(&handle->header_lock);
}

int _starpu_prefetch_task_input_on_node(struct starpu_task *task, uint32_t node)
{
	starpu_buffer_descr *descrs = task->buffers;
	unsigned nbuffers = task->cl->nbuffers;

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle handle = descrs[index].handle;
		starpu_access_mode mode = descrs[index].mode;

		if (mode & STARPU_SCRATCH)
			continue;

		struct starpu_data_replicate_s *replicate = handle->per_node[node];
		prefetch_data_on_node(handle, replicate, mode);

		_starpu_set_data_requested_flag_if_needed(replicate);
	}

	return 0;
}

int _starpu_fetch_task_input(struct starpu_task *task, uint32_t mask)
{
	STARPU_TRACE_START_FETCH_INPUT(NULL);

	uint32_t local_memory_node = _starpu_get_local_memory_node();

	starpu_buffer_descr *descrs = task->buffers;
	unsigned nbuffers = task->cl->nbuffers;

	/* TODO get that from the stack */
	starpu_job_t j = (struct starpu_job_s *)task->starpu_private;

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		int ret;
		starpu_data_handle handle = descrs[index].handle;
		starpu_access_mode mode = descrs[index].mode;

		void *interface;

		if (mode & STARPU_SCRATCH)
		{
			starpu_mem_chunk_t mc;
			mc = _starpu_memchunk_cache_lookup(local_memory_node, handle);
			if (!mc)
			{
				/* Cache miss */

				/* This is a scratch memory, so we duplicate (any of)
				 * the interface which contains sufficient information
				 * to allocate the buffer. */
				size_t interface_size = handle->ops->interface_size;
				void *src_interface = starpu_data_get_interface_on_node(handle, local_memory_node);
	
				/* Pass the interface to StarPU so that the buffer can be allocated */
				_starpu_allocate_interface(handle, src_interface, local_memory_node);

				size_t size = _starpu_data_get_size(handle);
#warning TODO create a replicate struct here:
				mc = _starpu_memchunk_init(handle, size, src_interface, interface_size, 1);
			}

			interface = mc->interface;
			j->scratch_memchunks[index] = mc;
		}
		else {
			/* That's a "normal" buffer (R/W) */
			struct starpu_data_replicate_s *local_replicate;
			local_replicate = handle->per_node[local_memory_node];
			ret = fetch_data(handle, local_replicate, mode);
			if (STARPU_UNLIKELY(ret))
				goto enomem;

			interface = starpu_data_get_interface_on_node(handle, local_memory_node);
		}
		
		task->interface[index] = interface;
	}

	STARPU_TRACE_END_FETCH_INPUT(NULL);

	return 0;

enomem:
	/* try to unreference all the input that were successfully taken */
	/* XXX broken ... */
	_STARPU_DISP("something went wrong with buffer %u\n", index);
	//push_codelet_output(task, index, mask);
	_starpu_push_task_output(task, mask);
	return -1;
}

void _starpu_push_task_output(struct starpu_task *task, uint32_t mask)
{
	STARPU_TRACE_START_PUSH_OUTPUT(NULL);

        starpu_buffer_descr *descrs = task->buffers;
        unsigned nbuffers = task->cl->nbuffers;

	starpu_job_t j = (struct starpu_job_s *)task->starpu_private;

	uint32_t local_node = _starpu_get_local_memory_node();

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle handle = descrs[index].handle;
		starpu_access_mode mode = descrs[index].mode;

		if (mode & STARPU_SCRATCH)
		{
			_starpu_memchunk_cache_insert(local_node, j->scratch_memchunks[index]);
		}
		else {
			struct starpu_data_replicate_s *replicate = handle->per_node[local_node];
			_starpu_release_data_on_node(handle, mask, replicate);
			_starpu_release_data_enforce_sequential_consistency(task, handle);
		}
	}

	STARPU_TRACE_END_PUSH_OUTPUT(NULL);
}

/* NB : this value can only be an indication of the status of a data
	at some point, but there is no strong garantee ! */
unsigned _starpu_is_data_present_or_requested(starpu_data_handle handle, uint32_t node)
{
	unsigned ret = 0;

// XXX : this is just a hint, so we don't take the lock ...
//	pthread_spin_lock(&handle->header_lock);

	if (handle->per_node[node]->state != STARPU_INVALID 
		|| handle->per_node[node]->requested || handle->per_node[node]->request)
		ret = 1;

//	pthread_spin_unlock(&handle->header_lock);

	return ret;
}

