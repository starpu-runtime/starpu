/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018-2021  Federal University of Rio Grande do Sul (UFRGS)
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

#include <limits.h>
#include <math.h>

#include <common/config.h>
#include <datawizard/coherency.h>
#include <datawizard/copy_driver.h>
#include <datawizard/write_back.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/sort_data_handles.h>
#include <core/dependencies/data_concurrency.h>
#include <core/disk.h>
#include <profiling/profiling.h>
#include <core/task.h>
#include <starpu_scheduler.h>
#include <core/workers.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

static int link_supports_direct_transfers(starpu_data_handle_t handle, unsigned src_node, unsigned dst_node, unsigned *handling_node);
int _starpu_select_src_node(starpu_data_handle_t handle, unsigned destination)
{
	int src_node = -1;
	unsigned i;

	unsigned nnodes = starpu_memory_nodes_get_count();

	/* first find a valid copy, either a STARPU_OWNER or a STARPU_SHARED */
	unsigned node;

	size_t size = _starpu_data_get_size(handle);
	double cost = INFINITY;
	unsigned src_node_mask = 0;

	for (node = 0; node < nnodes; node++)
	{
		if (handle->per_node[node].state != STARPU_INVALID)
		{
			/* we found a copy ! */
			src_node_mask |= (1<<node);
		}
	}

	if (src_node_mask == 0 && handle->init_cl)
	{
		/* No copy yet, but applicationg told us how to build it.  */
		return -1;
	}

	/* we should have found at least one copy ! */
	STARPU_ASSERT_MSG(src_node_mask != 0, "The data for the handle %p is requested, but the handle does not have a valid value. Perhaps some initialization task is missing?", handle);

	/* Without knowing the size, we won't know the cost */
	if (!size)
		cost = 0;

	/* Check whether we have transfer cost for all nodes, if so, take the minimum */
	if (cost)
		for (i = 0; i < nnodes; i++)
		{
			if (src_node_mask & (1<<i))
			{
				double time;
				unsigned handling_node;

				/* Avoid indirect transfers */
				/* TODO: but with NVLink, that might be better than a "direct" transfer that actually goes through the Host! */
				if (!link_supports_direct_transfers(handle, i, destination, &handling_node))
					continue;

				time = starpu_transfer_predict(i, destination, size);
				if (_STARPU_IS_ZERO(time))
				{
					/* No estimation, will have to revert to dumb strategy */
					cost = 0.0;
					break;
				}
				else if (time < cost)
				{
					cost = time;
					src_node = i;
				}
			}
		}

	if (cost && src_node != -1)
	{
		/* Could estimate through cost, return that */
		STARPU_ASSERT(handle->per_node[src_node].allocated || handle->per_node[src_node].mapped != STARPU_UNMAPPED);
		STARPU_ASSERT(handle->per_node[src_node].initialized);
		return src_node;
	}

	int i_ram = -1;
	int i_gpu = -1;
	int i_disk = -1;

	/* Revert to dumb strategy: take RAM unless only a GPU has it */
	for (i = 0; i < nnodes; i++)
	{
		if (src_node_mask & (1<<i))
		{
			int (*can_copy)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, unsigned handling_node) = handle->ops->copy_methods->can_copy;
			/* Avoid transfers which the interface does not want */
			if (can_copy)
			{
				void *src_interface = handle->per_node[i].data_interface;
				void *dst_interface = handle->per_node[destination].data_interface;
				unsigned handling_node;

				if (!link_supports_direct_transfers(handle, i, destination, &handling_node))
				{
					/* Avoid through RAM if the interface does not want it */
					void *ram_interface = handle->per_node[STARPU_MAIN_RAM].data_interface;
					if ((!can_copy(src_interface, i, ram_interface, STARPU_MAIN_RAM, i)
					  && !can_copy(src_interface, i, ram_interface, STARPU_MAIN_RAM, STARPU_MAIN_RAM))
					 || (!can_copy(ram_interface, STARPU_MAIN_RAM, dst_interface, destination, STARPU_MAIN_RAM)
					  && !can_copy(ram_interface, STARPU_MAIN_RAM, dst_interface, destination, destination)))
						continue;
				}
			}

			/* however GPU are expensive sources, really !
			 * 	Unless peer transfer is supported (and it would then have been selected above).
			 * 	Other should be ok */

			if (starpu_node_get_kind(i) == STARPU_CPU_RAM ||
			    starpu_node_get_kind(i) == STARPU_MPI_MS_RAM)
				i_ram = i;
			else if (starpu_node_get_kind(i) == STARPU_DISK_RAM)
				i_disk = i;
			else
				i_gpu = i;
		}
	}

	/* we have to use cpu_ram in first */
	if (i_ram != -1)
		src_node = i_ram;
	else if (i_gpu != -1)
	/* otherwise a gpu */
		src_node = i_gpu;
	else
	/* no luck we have to use the disk memory */
		src_node = i_disk;

	STARPU_ASSERT(src_node != -1);
	STARPU_ASSERT(handle->per_node[src_node].allocated || handle->per_node[src_node].mapped != STARPU_UNMAPPED);
	STARPU_ASSERT(handle->per_node[src_node].initialized);
	return src_node;
}

/* this may be called once the data is fetched with header and STARPU_RW-lock hold */
void _starpu_update_data_state(starpu_data_handle_t handle,
			       struct _starpu_data_replicate *requesting_replicate,
			       enum starpu_data_access_mode mode)
{
	if (mode == STARPU_UNMAP)
	{
		/* Unmap request, invalidate */
		requesting_replicate->state = STARPU_INVALID;
		return;
	}

	/* There is nothing to do for relaxed coherency modes (scratch or
	 * reductions) */
	if (!(mode & STARPU_RW))
		return;

	unsigned nnodes = starpu_memory_nodes_get_count();

	/* the data is present now */
	unsigned requesting_node = requesting_replicate->memory_node;

	if (mode & STARPU_W)
	{
		/* the requesting node now has the only valid copy */
		unsigned node;
		for (node = 0; node < nnodes; node++)
		{
			if (requesting_replicate->mapped == (int) node
				&& !_starpu_node_needs_map_update(requesting_node))
				/* The mapped node will be kept up to date */
				continue;
			if (handle->per_node[node].mapped == (int) requesting_node
				&& !_starpu_node_needs_map_update(node))
				/* The mapping node will be kept up to date */
				continue;
			if (handle->per_node[node].state != STARPU_INVALID)
				_starpu_trace_data_state_invalid(&handle, node);
			handle->per_node[node].state = STARPU_INVALID;
		}
		if (requesting_replicate->state != STARPU_OWNER)
			_starpu_trace_data_state_owner(&handle, requesting_node);
		requesting_replicate->state = STARPU_OWNER;
		if (handle->home_node != -1 && handle->per_node[handle->home_node].state == STARPU_INVALID)
			/* Notify that this MC is now dirty */
			_starpu_memchunk_dirty(requesting_replicate->mc, requesting_replicate->memory_node);
	}
	else
	{
		/* read only */
		if (requesting_replicate->state != STARPU_OWNER)
		{
			/* there was at least another copy of the data */
			unsigned node;
			for (node = 0; node < nnodes; node++)
			{
				struct _starpu_data_replicate *replicate = &handle->per_node[node];
				if (replicate->state != STARPU_INVALID)
				{
					if (replicate->state != STARPU_SHARED)
						_starpu_trace_data_state_shared(&handle, node);
					replicate->state = STARPU_SHARED;
				}
			}
			if (requesting_replicate->state != STARPU_SHARED)
				_starpu_trace_data_state_shared(&handle, requesting_node);
			requesting_replicate->state = STARPU_SHARED;
		}
	}
}

static int worker_supports_direct_access(unsigned node, unsigned handling_node)
{
	if (node == handling_node)
		return 1;

	if (!_starpu_memory_node_get_nworkers(handling_node))
		/* No worker to process the request from that node */
		return 0;

	const struct _starpu_node_ops *node_ops = _starpu_memory_node_get_node_ops(node);
	if (node_ops && node_ops->is_direct_access_supported)
		return node_ops->is_direct_access_supported(node, handling_node);
	else
		return 0;
}

static int link_supports_direct_transfers(starpu_data_handle_t handle, unsigned src_node, unsigned dst_node, unsigned *handling_node)
{
	STARPU_ASSERT_MSG(handle->ops->copy_methods, "The handle %s does not define a copy_methods\n", handle->ops->name);
	int (*can_copy)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, unsigned handling_node) = handle->ops->copy_methods->can_copy;
	void *src_interface = handle->per_node[src_node].data_interface;
	void *dst_interface = handle->per_node[dst_node].data_interface;

	/* Note: with CUDA, performance seems a bit better when issuing the transfer from the destination (tested without GPUDirect, but GPUDirect probably behave the same) */
	if (worker_supports_direct_access(src_node, dst_node) && (!can_copy || can_copy(src_interface, src_node, dst_interface, dst_node, dst_node)))
	{
		*handling_node = dst_node;
		return 1;
	}

	if (worker_supports_direct_access(dst_node, src_node) && (!can_copy || can_copy(src_interface, src_node, dst_interface, dst_node, src_node)))
	{
		*handling_node = src_node;
		return 1;
	}

	return 0;
}

/* Now, we use slowness/bandwidth to compare numa nodes, is it better to use latency ? */
static unsigned chose_best_numa_between_src_and_dest(int src, int dst)
{
	double timing_best;
	int best_numa = -1;
	unsigned numa;
	const unsigned nb_numa_nodes = starpu_memory_nodes_get_numa_count();
	for(numa = 0; numa < nb_numa_nodes; numa++)
	{
		double actual = 1.0/starpu_transfer_bandwidth(src, numa) + 1.0/starpu_transfer_bandwidth(numa, dst);

		/* Compare slowness : take the lowest */
		if (best_numa < 0 || actual < timing_best)
		{
			best_numa = numa;
			timing_best = actual;
		}
	}
	STARPU_ASSERT(best_numa >= 0);

	return best_numa;
}

/* Determines the path of a request : each hop is defined by (src,dst) and the
 * node that handles the hop. The returned value indicates the number of hops,
 * and the max_len is the maximum number of hops (ie. the size of the
 * src_nodes, dst_nodes and handling_nodes arrays. */
int _starpu_determine_request_path(starpu_data_handle_t handle,
				  int src_node, int dst_node,
				  enum starpu_data_access_mode mode, int max_len,
				  unsigned *src_nodes, unsigned *dst_nodes,
				  unsigned *handling_nodes, unsigned write_invalidation)
{
	if ((mode & STARPU_R) && src_node >= 0 && dst_node >= 0)
	{

		struct _starpu_data_replicate *src_replicate = &handle->per_node[src_node];
		struct _starpu_data_replicate *dst_replicate = &handle->per_node[dst_node];

		if (src_replicate->mapped != STARPU_UNMAPPED)
		{
			/* Device -> map */
			STARPU_ASSERT(max_len >= 1);
			*src_nodes++ = src_node;
			*dst_nodes++ = src_replicate->mapped;
			*handling_nodes++ = src_node;
			max_len--;

			/* map -> Device */
			int consumed = _starpu_determine_request_path(handle,
					src_replicate->mapped, dst_node,
					mode, max_len,
					src_nodes, dst_nodes, handling_nodes,
					write_invalidation);

			return consumed + 1;
		}
		else if (dst_replicate->mapped != STARPU_UNMAPPED)
		{
			/* Device -> map */
			int consumed = _starpu_determine_request_path(handle,
					src_node, dst_replicate->mapped,
					mode, max_len,
					src_nodes, dst_nodes, handling_nodes,
					write_invalidation);

			src_nodes += consumed;
			dst_nodes += consumed;
			handling_nodes += consumed;
			max_len -= consumed;

			/* map -> Device */
			STARPU_ASSERT(max_len >= 1);
			*src_nodes++ = dst_replicate->mapped;
			*dst_nodes++ = dst_node;
			*handling_nodes++ = dst_node;
			max_len--;

			return consumed + 1;
		}
	}

	if (src_node == dst_node || !(mode & STARPU_R))
	{
		if (dst_node == -1 || starpu_node_get_kind(dst_node) == STARPU_DISK_RAM)
			handling_nodes[0] = src_node;
		else
			handling_nodes[0] = dst_node;

		if (write_invalidation)
			/* The invalidation request will be enough */
			return 0;

		/* The destination node should only allocate the data, no transfer is required */
		STARPU_ASSERT(max_len >= 1);
		src_nodes[0] = dst_node; // ignored
		dst_nodes[0] = dst_node;
		return 1;
	}

	if (src_node < 0)
	{
		/* Will just initialize the destination */
		STARPU_ASSERT(max_len >= 1);
		src_nodes[0] = dst_node; // ignored
		dst_nodes[0] = dst_node;
		return 1;
	}

	unsigned handling_node;
	int link_is_valid = link_supports_direct_transfers(handle, src_node, dst_node, &handling_node);

	if (!link_is_valid)
	{
		int (*can_copy)(void *, unsigned, void *, unsigned, unsigned) = handle->ops->copy_methods->can_copy;
		void *src_interface = handle->per_node[src_node].data_interface;
		void *dst_interface = handle->per_node[dst_node].data_interface;

		/* We need an intermediate hop to implement data staging
		 * through main memory. */
		STARPU_ASSERT(max_len >= 2);
		STARPU_ASSERT(src_node >= 0);

		unsigned numa = chose_best_numa_between_src_and_dest(src_node, dst_node);

		/* GPU -> RAM */
		src_nodes[0] = src_node;
		dst_nodes[0] = numa;

		if (starpu_node_get_kind(src_node) == STARPU_DISK_RAM)
			/* Disks don't have their own driver thread */
			handling_nodes[0] = dst_node;
		else if (!can_copy || can_copy(src_interface, src_node, dst_interface, dst_node, src_node))
		{
			handling_nodes[0] = src_node;
		}
		else
		{
			STARPU_ASSERT_MSG(can_copy(src_interface, src_node, dst_interface, dst_node, dst_node), "interface %d refuses all kinds of transfers from node %d to node %d\n", handle->ops->interfaceid, src_node, dst_node);
			handling_nodes[0] = dst_node;
		}

		/* RAM -> GPU */
		src_nodes[1] = numa;
		dst_nodes[1] = dst_node;

		if (starpu_node_get_kind(dst_node) == STARPU_DISK_RAM)
			/* Disks don't have their own driver thread */
			handling_nodes[1] = src_node;
		else if (!can_copy || can_copy(src_interface, src_node, dst_interface, dst_node, dst_node))
		{
			handling_nodes[1] = dst_node;
		}
		else
		{
			STARPU_ASSERT_MSG(can_copy(src_interface, src_node, dst_interface, dst_node, src_node), "interface %d refuses all kinds of transfers from node %d to node %d\n", handle->ops->interfaceid, src_node, dst_node);
			handling_nodes[1] = src_node;
		}

		return 2;
	}
	else
	{
		STARPU_ASSERT(max_len >= 1);

		src_nodes[0] = src_node;
		dst_nodes[0] = dst_node;
		handling_nodes[0] = handling_node;

		return 1;
	}
}

/* handle->lock should be taken. r is returned locked. The node parameter
 * indicate either the source of the request, or the destination for a
 * write-only request. */
static struct _starpu_data_request *_starpu_search_existing_data_request(struct _starpu_data_replicate *replicate, unsigned node, enum starpu_data_access_mode mode, struct starpu_task *task, enum starpu_is_prefetch is_prefetch)
{
	struct _starpu_data_request *r;

	/* Make sure we don't have anything else than R/W */
	STARPU_ASSERT(mode != STARPU_UNMAP);

	for (r = replicate->request[node]; r; r = r->next_same_req)
	{
		_starpu_spin_checklocked(&r->handle->header_lock);

		if (r->canceled)
			/* Do not reuse a cancelled request */
			continue;

		if (task && r->task && task != r->task)
			/* Do not collapse requests for different tasks */
			continue;

		_starpu_spin_lock(&r->lock);

		/* perhaps we need to "upgrade" the request */
		if (is_prefetch < r->prefetch)
			_starpu_update_prefetch_status(r, is_prefetch);

		/* TODO: abort on unmapping request */

		if (mode & STARPU_R)
		{
			/* in case the existing request did not imply a memory
			 * transfer yet, we have to take a second refcnt now
			 * for the source, in addition to the refcnt for the
			 * destination
			 * (so that the source remains valid) */
			if (!(r->mode & STARPU_R))
			{
				replicate->refcnt++;
				replicate->handle->busy_count++;
			}

			r->mode = (enum starpu_data_access_mode) ((int) r->mode | (int) STARPU_R);
		}

		if (mode & STARPU_W)
			r->mode = (enum starpu_data_access_mode) ((int) r->mode | (int)  STARPU_W);

		/* We collapse with this request */
		return r;
	}

	return NULL;
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

struct _starpu_data_request *_starpu_create_request_to_fetch_data(starpu_data_handle_t handle,
								  struct _starpu_data_replicate *dst_replicate,
								  enum starpu_data_access_mode mode,
								  struct starpu_task *task,
								  enum starpu_is_prefetch is_prefetch,
								  unsigned async,
								  void (*callback_func)(void *), void *callback_arg, int prio, const char *origin)
{
	enum starpu_data_access_mode orig_mode = mode;

	/* We don't care about commuting for data requests, that was handled before. */
	mode &= ~STARPU_COMMUTE;

	/* This function is called with handle's header lock taken */
	_starpu_spin_checklocked(&handle->header_lock);

	/* TODO: If writing copying to RAM, first update maps into RAM, even if RAM is already up to date.  */

	int requesting_node = dst_replicate ? dst_replicate->memory_node : -1;
	unsigned nwait = 0;

	if (mode & STARPU_W)
	{
		/* We will write to the buffer. We will have to wait for all
		 * existing requests before the last request which will
		 * invalidate all their results (which were possibly spurious,
		 * e.g. too aggressive eviction).
		 */
		unsigned i, j;
		unsigned nnodes = starpu_memory_nodes_get_count();
		for (i = 0; i < nnodes; i++)
			for (j = 0; j < nnodes; j++)
			{
				struct _starpu_data_request *r;
				for (r = handle->per_node[i].request[j]; r; r = r->next_same_req)
					nwait++;
			}
		/* If the request is not detached (i.e. the caller really wants
		 * proper ownership), no new requests will appear because a
		 * reference will be kept on the dst replicate, which will
		 * notably prevent data reclaiming.
		 */
	}

	if ((!dst_replicate || dst_replicate->state != STARPU_INVALID) && (!nwait || is_prefetch))
	{
		if (dst_replicate)
		{
#ifdef STARPU_MEMORY_STATS
			enum _starpu_cache_state old_state = dst_replicate->state;
#endif
			/* the data is already available and we don't have to wait for
			 * any request, so we can stop */
			_starpu_update_data_state(handle, dst_replicate, mode);
			_starpu_msi_cache_hit(requesting_node);

#ifdef STARPU_MEMORY_STATS
			_starpu_memory_handle_stats_cache_hit(handle, requesting_node);

			/* XXX Broken ? */
			if (old_state == STARPU_SHARED
			    && dst_replicate->state == STARPU_OWNER)
				_starpu_memory_handle_stats_shared_to_owner(handle, requesting_node);
#endif

			if (dst_replicate->mc)
			{
				if (is_prefetch == STARPU_TASK_PREFETCH)
					/* Make sure it stays there */
					dst_replicate->nb_tasks_prefetch++;

				_starpu_memchunk_recently_used(dst_replicate->mc, requesting_node);
			}

			if (task)
			{
				unsigned j;
				unsigned nnodes = starpu_memory_nodes_get_count();
				/* Cancel any existing (prefetch) request */
				struct _starpu_data_request *r2;
				for (j = 0; j < nnodes; j++)
				{
					for (r2 = dst_replicate->request[j]; r2; r2 = r2->next_same_req)
					{
						if (r2->task && r2->task == task)
						{
							r2->canceled = 1;
							break;
						}
					}
				}
			}
		}

		_starpu_spin_unlock(&handle->header_lock);

		if (callback_func)
			callback_func(callback_arg);

		_STARPU_LOG_OUT_TAG("data available");
		return NULL;
	}

	if (dst_replicate)
		_starpu_msi_cache_miss(requesting_node);

	/* the only remaining situation is that the local copy was invalid */
	STARPU_ASSERT((dst_replicate && dst_replicate->state == STARPU_INVALID) || nwait);

	/* find someone who already has the data */
	int src_node = -1;

	if (dst_replicate && mode & STARPU_R)
	{
		if (dst_replicate->state == STARPU_INVALID)
			src_node = _starpu_select_src_node(handle, requesting_node);
		else
			src_node = requesting_node;
		if (src_node < 0)
		{
			/* We will create it, no need to read an existing value */
			mode &= ~STARPU_R;
		}
	}
	if (dst_replicate && !(mode & STARPU_R))
	{
		/* if the data is in write only mode (and not SCRATCH or REDUX), there is no need for a source, data will be initialized by the task itself */
		if (mode & STARPU_W && is_prefetch <= STARPU_TASK_PREFETCH)
		{
			if (orig_mode & STARPU_R)
				/* _starpu_fetch_task_input_tail will call init_cl */
				STARPU_ASSERT(handle->init_cl);
			else
				/* No initialization needed at all */
				dst_replicate->initialized = 1;
		}
		if (starpu_node_get_kind(requesting_node) == STARPU_CPU_RAM && !nwait
			&& !_starpu_malloc_willpin_on_node(requesting_node))
		{
			/* FIXME: also try to map */
			/* And this is the main RAM without pinning, really no need for a
			 * request, just quickly allocate and be done */
			if (dst_replicate->mapped != STARPU_UNMAPPED
				|| _starpu_allocate_memory_on_node(handle, dst_replicate, is_prefetch, 0) == 0)
			{
				if (is_prefetch <= STARPU_TASK_PREFETCH)
					_starpu_update_data_state(handle, dst_replicate, mode);
				if (dst_replicate->mc)
				{
					if (is_prefetch == STARPU_TASK_PREFETCH)
						/* Make sure it stays there */
						dst_replicate->nb_tasks_prefetch++;

					_starpu_memchunk_recently_used(dst_replicate->mc, requesting_node);
				}

				_starpu_spin_unlock(&handle->header_lock);

				if (callback_func)
					callback_func(callback_arg);
				_STARPU_LOG_OUT_TAG("data immediately allocated");
				return NULL;
			}
		}
	}

#define MAX_REQUESTS 4
	/* We can safely assume that there won't be more than 2 hops in the
	 * current implementation */
	unsigned src_nodes[MAX_REQUESTS], dst_nodes[MAX_REQUESTS], handling_nodes[MAX_REQUESTS];
	/* keep one slot for the last W request, if any */
	int write_invalidation = (mode & STARPU_W) && nwait && !is_prefetch;
	int nhops = _starpu_determine_request_path(handle, src_node, requesting_node, mode, MAX_REQUESTS,
					   src_nodes, dst_nodes, handling_nodes, write_invalidation);

	STARPU_ASSERT(nhops >= 0 && nhops <= MAX_REQUESTS-1);
	struct _starpu_data_request *requests[nhops + write_invalidation];

	/* Did we reuse a request for that hop ? */
	int reused_requests[nhops + write_invalidation];

	/* Construct an array with a list of requests, possibly reusing existing requests */
	int hop;
	for (hop = 0; hop < nhops; hop++)
	{
		struct _starpu_data_request *r;

		unsigned hop_src_node = src_nodes[hop];
		unsigned hop_dst_node = dst_nodes[hop];
		unsigned hop_handling_node = handling_nodes[hop];

		struct _starpu_data_replicate *hop_src_replicate;
		struct _starpu_data_replicate *hop_dst_replicate;

		/* Only the first request is independent */
		unsigned ndeps = (hop == 0)?0:1;

		hop_src_replicate = &handle->per_node[hop_src_node];
		hop_dst_replicate = (hop != nhops - 1)?&handle->per_node[hop_dst_node]:dst_replicate;

		/* Try to reuse a request if possible */
#ifdef STARPU_DEVEL
#warning We do not actually want to reuse an existing request when our request is for a task with low priority, that will get executed much later. We don t want to wire down the data in between, at worse that could hog the complete gpu memory...
#endif
		r = _starpu_search_existing_data_request(hop_dst_replicate,
							 (mode & STARPU_R)?hop_src_node:hop_dst_node,
							 mode, task, is_prefetch);

		reused_requests[hop] = !!r;

		if (!r)
		{
			/* Create a new request if there was no request to reuse */
			r = _starpu_create_data_request(handle, hop_src_replicate,
							hop_dst_replicate, hop_handling_node,
							mode, ndeps, task, is_prefetch, prio, 0, origin);
			nwait++;
		}

		requests[hop] = r;
	}

	/* Chain these requests */
	for (hop = 0; hop < nhops; hop++)
	{
		struct _starpu_data_request *r;
		r = requests[hop];

		if (hop != nhops - 1)
		{
			if (!reused_requests[hop + 1])
			{
				r->next_req[r->next_req_count++] = requests[hop + 1];
				STARPU_ASSERT(r->next_req_count <= STARPU_MAXNODES);
			}
		}
		else
		{
			if (is_prefetch == STARPU_TASK_PREFETCH)
				/* Make last request add the prefetch count on the mc to keep the data
				 * there until the task gets to execute.  */
				r->nb_tasks_prefetch++;

			if (!write_invalidation)
				/* The last request will perform the callback after termination */
				_starpu_data_request_append_callback(r, callback_func, callback_arg);
		}

		if (reused_requests[hop])
			_starpu_spin_unlock(&r->lock);
	}

	if (write_invalidation)
	{
		/* Some requests were still pending, we have to add yet another
		 * request, depending on them, which will invalidate their
		 * result.
		 */
		struct _starpu_data_request *r = _starpu_create_data_request(handle, dst_replicate,
							dst_replicate, requesting_node,
							STARPU_W, nwait, task, is_prefetch, prio, 1, origin);

		/* and perform the callback after termination */
		_starpu_data_request_append_callback(r, callback_func, callback_arg);

		/* We will write to the buffer. We will have to wait for all
		 * existing requests before the last request which will
		 * invalidate all their results (which were possibly spurious,
		 * e.g. too aggressive eviction).
		 */
		unsigned i, j;
		unsigned nnodes = starpu_memory_nodes_get_count();
		for (i = 0; i < nnodes; i++)
			for (j = 0; j < nnodes; j++)
			{
				struct _starpu_data_request *r2;
				for (r2 = handle->per_node[i].request[j]; r2; r2 = r2->next_same_req)
				{
					_starpu_spin_lock(&r2->lock);
					if (is_prefetch < r2->prefetch)
						/* Hasten the request we will have to wait for */
						_starpu_update_prefetch_status(r2, is_prefetch);
					r2->next_req[r2->next_req_count++] = r;
					STARPU_ASSERT(r2->next_req_count <= STARPU_MAXNODES + 1);
					_starpu_spin_unlock(&r2->lock);
					nwait--;
				}
			}
		STARPU_ASSERT(nwait == 0);

		nhops++;
		requests[nhops - 1] = r;
		/* existing requests will post this one */
		reused_requests[nhops - 1] = 1;
	}
	STARPU_ASSERT(nhops);

	if (!async)
		requests[nhops - 1]->refcnt++;


	/* we only submit the first request, the remaining will be
	 * automatically submitted afterward */
	if (!reused_requests[0])
		_starpu_post_data_request(requests[0]);

	return requests[nhops - 1];
}

int _starpu_fetch_data_on_node(starpu_data_handle_t handle, int node, struct _starpu_data_replicate *dst_replicate,
			       enum starpu_data_access_mode mode, unsigned detached,
			       struct starpu_task *task, enum starpu_is_prefetch is_prefetch, unsigned async,
			       void (*callback_func)(void *), void *callback_arg, int prio, const char *origin)
{
	_STARPU_LOG_IN();

	_starpu_spin_lock(&handle->header_lock);

	if (mode & STARPU_R && is_prefetch > STARPU_FETCH)
	{
		unsigned src_node_mask = 0;

		unsigned nnodes = starpu_memory_nodes_get_count();
		unsigned n;
		for (n = 0; n < nnodes; n++)
		{
			if (handle->per_node[n].state != STARPU_INVALID)
			{
				/* we found a copy ! */
				src_node_mask |= (1<<n);
			}
		}

		if (src_node_mask == 0)
		{
			/* no valid copy, nothing to prefetch */
			STARPU_ASSERT_MSG(handle->init_cl, "Could not find a valid copy of the data, and no handle initialization function");
			_starpu_spin_unlock(&handle->header_lock);
			return 0;
		}
	}

	if (!detached)
	{
		/* Take references which will be released by _starpu_release_data_on_node */
		if (dst_replicate)
			dst_replicate->refcnt++;
		else if (node == STARPU_ACQUIRE_NO_NODE_LOCK_ALL)
		{
			int i;
			for (i = 0; i < STARPU_MAXNODES; i++)
				handle->per_node[i].refcnt++;
		}
		handle->busy_count++;
	}

	struct _starpu_data_request *r;
	r = _starpu_create_request_to_fetch_data(handle, dst_replicate, mode,
						 task, is_prefetch, async, callback_func, callback_arg, prio, origin);

	/* If no request was created, the handle was already up-to-date on the
	 * node. In this case, _starpu_create_request_to_fetch_data has already
	 * unlocked the header. */
	if (!r)
		return 0;

	_starpu_spin_unlock(&handle->header_lock);

	int ret = async?0:_starpu_wait_data_request_completion(r, 1);
	_STARPU_LOG_OUT();
	return ret;
}

static int idle_prefetch_data_on_node(starpu_data_handle_t handle, int node, struct _starpu_data_replicate *replicate, enum starpu_data_access_mode mode, struct starpu_task *task, int prio)
{
	return _starpu_fetch_data_on_node(handle, node, replicate, mode, 1, task, STARPU_IDLEFETCH, 1, NULL, NULL, prio, "idle_prefetch_data_on_node");
}

static int task_prefetch_data_on_node(starpu_data_handle_t handle, int node, struct _starpu_data_replicate *replicate, enum starpu_data_access_mode mode, struct starpu_task *task, int prio)
{
	return _starpu_fetch_data_on_node(handle, node, replicate, mode, 1, task, STARPU_TASK_PREFETCH, 1, NULL, NULL, prio, "task_prefetch_data_on_node");
}

static int STARPU_ATTRIBUTE_UNUSED prefetch_data_on_node(starpu_data_handle_t handle, int node, struct _starpu_data_replicate *replicate, enum starpu_data_access_mode mode, struct starpu_task *task, int prio)
{
	return _starpu_fetch_data_on_node(handle, node, replicate, mode, 1, task, STARPU_PREFETCH, 1, NULL, NULL, prio, "prefetch_data_on_node");
}

static int fetch_data(starpu_data_handle_t handle, int node, struct _starpu_data_replicate *replicate, enum starpu_data_access_mode mode, struct starpu_task *task, int prio)
{
	return _starpu_fetch_data_on_node(handle, node, replicate, mode, 0, task, STARPU_FETCH, 0, NULL, NULL, prio, "fetch_data");
}

uint32_t _starpu_get_data_refcnt(starpu_data_handle_t handle, unsigned node)
{
	return handle->per_node[node].refcnt;
}

size_t _starpu_data_get_size(starpu_data_handle_t handle)
{
	return handle->ops->get_size(handle);
}

size_t _starpu_data_get_alloc_size(starpu_data_handle_t handle)
{
	if (handle->ops->get_alloc_size)
		return handle->ops->get_alloc_size(handle);
	else
		return handle->ops->get_size(handle);
}

starpu_ssize_t _starpu_data_get_max_size(starpu_data_handle_t handle)
{
	if (handle->ops->get_max_size)
		return handle->ops->get_max_size(handle);
	else
		return -1;
}

uint32_t _starpu_data_get_footprint(starpu_data_handle_t handle)
{
	return handle->footprint;
}

/* in case the data was accessed on a write mode, do not forget to
 * make it accessible again once it is possible ! */
void _starpu_release_data_on_node(starpu_data_handle_t handle, uint32_t default_wt_mask, enum starpu_data_access_mode down_to_mode, struct _starpu_data_replicate *replicate)
{
	uint32_t wt_mask;
	size_t max_wt_mask = sizeof(wt_mask) * 8;
	unsigned wt_count = starpu_memory_nodes_get_count();
	if (max_wt_mask > STARPU_MAXNODES)
		max_wt_mask = STARPU_MAXNODES;
	if (wt_count > max_wt_mask)
		wt_count = max_wt_mask;

	wt_mask = default_wt_mask | handle->wt_mask;
	wt_mask &= (1ULL<<max_wt_mask)-1;

	/* Note that it is possible that there is no valid copy of the data (if
	 * starpu_data_invalidate/deinitialize was called for instance). In that case, we do
	 * not enforce any write-through mechanism. */

	unsigned memory_node = replicate->memory_node;

	if (replicate->state != STARPU_INVALID && handle->current_mode & STARPU_W)
	if (wt_mask && (memory_node >= max_wt_mask || wt_mask & ~(1<<memory_node)))
		_starpu_write_through_data(handle, memory_node, wt_mask);

	int cpt = 0;
	while (cpt < STARPU_SPIN_MAXTRY && _starpu_spin_trylock(&handle->header_lock))
	{
		cpt++;
		_starpu_datawizard_progress(_STARPU_DATAWIZARD_DO_ALLOC);
	}
	if (cpt == STARPU_SPIN_MAXTRY)
		_starpu_spin_lock(&handle->header_lock);

	if (down_to_mode == STARPU_NONE)
	{
		/* Release refcnt taken by fetch_data_on_node */
		replicate->refcnt--;
		STARPU_ASSERT_MSG(replicate->refcnt >= 0, "handle %p released too many times on node %u", handle, memory_node);

		STARPU_ASSERT_MSG(handle->busy_count > 0, "handle %p released too many times on node %u", handle, memory_node);
		handle->busy_count--;
	}

	if (!_starpu_notify_data_dependencies(handle, down_to_mode))
		_starpu_spin_unlock(&handle->header_lock);
}

int _starpu_prefetch_task_input_prio(struct starpu_task *task, int target_node, int worker, int prio, enum starpu_is_prefetch prefetch)
{
#ifdef STARPU_OPENMP
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	/* do not attempt to prefetch task input if this is an OpenMP task resuming after blocking */
	if (j->discontinuous != 0)
		return 0;
#endif
	STARPU_ASSERT_MSG(prefetch != STARPU_PREFETCH || !task->prefetched, "Prefetching was already requested for this task! Did you set 'prefetches' to 1 in the starpu_sched_policy structure?");
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	unsigned index;

	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, index);
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, index);

		if (mode & (STARPU_SCRATCH|STARPU_REDUX))
			continue;

		int node;
		if (target_node >= 0)
			node = _starpu_task_data_get_node_on_node(task, index, target_node);
		else
			node = _starpu_task_data_get_node_on_worker(task, index, worker);

		if (node < 0)
			continue;

		struct _starpu_data_replicate *replicate = &handle->per_node[node];
		if (prefetch == STARPU_PREFETCH)
			task_prefetch_data_on_node(handle, node, replicate, mode, task, prio);
		else
			idle_prefetch_data_on_node(handle, node, replicate, mode, task, prio);
	}

	if (prefetch == STARPU_PREFETCH)
		task->prefetched = 1;

	return 0;
}

int starpu_prefetch_task_input_prio(struct starpu_task *task, int target_node, int worker, int prio)
{
	return _starpu_prefetch_task_input_prio(task, target_node, worker, prio, STARPU_PREFETCH);
}

int starpu_prefetch_task_input_on_node_prio(struct starpu_task *task, unsigned target_node, int prio)
{
	return starpu_prefetch_task_input_prio(task, target_node, -1, prio);
}


int starpu_prefetch_task_input_on_node(struct starpu_task *task, unsigned node)
{
	int prio = task->priority;
	if (task->workerorder)
		prio = INT_MAX - task->workerorder;
	return starpu_prefetch_task_input_on_node_prio(task, node, prio);
}

int starpu_idle_prefetch_task_input_prio(struct starpu_task *task, int target_node, int worker, int prio)
{
	return _starpu_prefetch_task_input_prio(task, target_node, worker, prio, STARPU_IDLEFETCH);
}

int starpu_idle_prefetch_task_input_on_node_prio(struct starpu_task *task, unsigned target_node, int prio)
{
	return starpu_idle_prefetch_task_input_prio(task, target_node, -1, prio);
}

int starpu_idle_prefetch_task_input_on_node(struct starpu_task *task, unsigned node)
{
	int prio = task->priority;
	if (task->workerorder)
		prio = INT_MAX - task->workerorder;
	return starpu_idle_prefetch_task_input_on_node_prio(task, node, prio);
}

int starpu_prefetch_task_input_for_prio(struct starpu_task *task, unsigned worker, int prio)
{
	return starpu_prefetch_task_input_prio(task, -1, worker, prio);
}

int starpu_prefetch_task_input_for(struct starpu_task *task, unsigned worker)
{
	int prio = task->priority;
	if (task->workerorder)
		prio = INT_MAX - task->workerorder;
	return starpu_prefetch_task_input_for_prio(task, worker, prio);
}

int starpu_idle_prefetch_task_input_for_prio(struct starpu_task *task, unsigned worker, int prio)
{
	return starpu_idle_prefetch_task_input_prio(task, -1, worker, prio);
}

int starpu_idle_prefetch_task_input_for(struct starpu_task *task, unsigned worker)
{
	int prio = task->priority;
	if (task->workerorder)
		prio = INT_MAX - task->workerorder;
	return starpu_idle_prefetch_task_input_for_prio(task, worker, prio);
}

static struct _starpu_data_replicate *get_replicate(starpu_data_handle_t handle, enum starpu_data_access_mode mode, int workerid, unsigned node)
{
	if (mode & (STARPU_SCRATCH|STARPU_REDUX))
	{
		STARPU_ASSERT(workerid >= 0);
		if (STARPU_RUNNING_ON_VALGRIND || !handle->per_worker)
		{
			_starpu_spin_lock(&handle->header_lock);
			if (!handle->per_worker)
				_starpu_data_initialize_per_worker(handle);
			_starpu_spin_unlock(&handle->header_lock);
		}
		return &handle->per_worker[workerid];
	}
	else
		/* That's a "normal" buffer (R/W) */
		return &handle->per_node[node];
}

/* Callback used when a buffer is send asynchronously to the sink */
static void _starpu_fetch_task_input_cb(void *arg)
{
	struct _starpu_worker * worker = (struct _starpu_worker *) arg;

	/* increase the number of buffer received */
	STARPU_WMB();
	(void)STARPU_ATOMIC_ADD(&worker->nb_buffers_transferred, 1);

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_broadcast(&_starpu_simgrid_transfer_queue[worker->memory_node]);
#endif
}

/* Synchronously or asynchronously fetch data for a given task (if it's not there already)
 * Returns the number of data acquired here.  */

/* _starpu_fetch_task_input must be called before
 * executing the task. __starpu_push_task_output but be called after the
 * execution of the task. */

/* The driver can either just call _starpu_fetch_task_input with async==0,
 * or to improve overlapping, it can call _starpu_fetch_task_input with
 * async==1, then wait for transfers to complete, then call
 * _starpu_fetch_task_input_tail to complete the fetch.	 */
int _starpu_fetch_task_input(struct starpu_task *task, struct _starpu_job *j, int async)
{
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	int workerid = worker->workerid;
	if (async)
	{
		worker->task_transferring = task;
		worker->nb_buffers_transferred = 0;
		if (worker->ntasks <= 1)
			_starpu_trace_worker_start_fetch_input(NULL, workerid);
	}
	else
		_starpu_trace_start_fetch_input(NULL);

	int profiling = starpu_profiling_status_get();
	if (profiling && task->profiling_info)
		_starpu_clock_gettime(&task->profiling_info->acquire_data_start_time);

	struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	unsigned nacquires;

	unsigned index;
	int indexdup;

	nacquires = 0;
	for (index = 0; index < nbuffers; index++)
	{
		int ret;
		starpu_data_handle_t handle = descrs[index].handle;
		enum starpu_data_access_mode mode = descrs[index].mode;
		int orig_node = descrs[index].orig_node;
		int node = _starpu_task_data_get_node_on_worker(task, descrs[index].index, workerid);
		/* We set this here for coherency with __starpu_push_task_output */
		descrs[index].node = node;
		if (mode == STARPU_NONE ||
			(mode & ((1<<STARPU_MODE_SHIFT) - 1)) >= STARPU_ACCESS_MODE_MAX ||
			(mode >> STARPU_MODE_SHIFT) >= (STARPU_SHIFTED_MODE_MAX >> STARPU_MODE_SHIFT))
			STARPU_ASSERT_MSG(0, "mode %d (0x%x) is bogus\n", mode, mode);

		if (node < 0)
			continue;

		struct _starpu_data_replicate *local_replicate;

		for (indexdup = (int) index-1; indexdup >= 0; indexdup--)
		{
			starpu_data_handle_t handle_dup = descrs[indexdup].handle;
			int node_dup = descrs[indexdup].orig_node;
			if (handle_dup == handle && node_dup == orig_node)
				/* We have already taken this data, skip it. This
				 * depends on ordering putting writes before reads, see
				 * _starpu_compar_handles */
				goto next;
			if (!_starpu_handles_same_root(handle_dup, handle))
				/* We are not checking within the same parent any more, no need to continue checking other handles */
				break;
		}

		local_replicate = get_replicate(handle, mode, workerid, node);

		if (async)
		{
			ret = _starpu_fetch_data_on_node(handle, node, local_replicate, mode, 0, task, STARPU_FETCH, 1,
					_starpu_fetch_task_input_cb, worker, task->priority, "_starpu_fetch_task_input");
#ifdef STARPU_SIMGRID
			if (_starpu_simgrid_fetching_input_cost())
				starpu_sleep(0.000001);
#endif
			if (STARPU_UNLIKELY(ret))
			{
				/* Ooops, not enough memory, make worker wait for these for now, and the synchronous call will finish by forcing eviction*/
				worker->nb_buffers_totransfer = nacquires;
				_starpu_add_worker_status(worker, STATUS_INDEX_WAITING, NULL);
				return 0;
			}
		}
		else
		{
			ret = fetch_data(handle, node, local_replicate, mode, task, task->priority);
#ifdef STARPU_SIMGRID
			if (_starpu_simgrid_fetching_input_cost())
				starpu_sleep(0.000001);
#endif
			if (STARPU_UNLIKELY(ret))
				goto enomem;
		}

		nacquires++;
	next:
		;
	}
	_starpu_add_worker_status(worker, STATUS_INDEX_WAITING, NULL);
	if (async)
	{
		worker->nb_buffers_totransfer = nacquires;
		return 0;
	}

	_starpu_fetch_task_input_tail(task, j, worker);

	return 0;

enomem:
	_starpu_trace_end_fetch_input(NULL);
	_STARPU_DISP("something went wrong with buffer %u\n", index);

	/* try to unreference all the input that were successfully taken */
	unsigned index2;
	for (index2 = 0; index2 < index; index2++)
	{
		starpu_data_handle_t handle = descrs[index2].handle;
		enum starpu_data_access_mode mode = descrs[index2].mode;
		int orig_node = descrs[index2].orig_node;
		int node = descrs[index2].node;

		struct _starpu_data_replicate *local_replicate;

		for (indexdup = (int) index2+1; indexdup < (int) index; indexdup++)
		{
			starpu_data_handle_t handle_dup = descrs[indexdup].handle;
			int node_dup = descrs[indexdup].orig_node;
			if (handle_dup == handle && node_dup == orig_node)
				/* We have already released this data, skip it. This
				 * depends on ordering putting writes before reads, see
				 * _starpu_compar_handles */
				goto next2;
			if (!_starpu_handles_same_root(handle_dup, handle))
				/* We are not checking within the same parent any more, no need to continue checking other handles */
				break;
		}

		local_replicate = get_replicate(handle, mode, workerid, node);

		_starpu_release_data_on_node(handle, 0, STARPU_NONE, local_replicate);
	next2:
		;
	}

	return -1;
}

/* Now that we have taken the data locks in locking order, fill the codelet interfaces in function order.  */
void _starpu_fetch_task_input_tail(struct starpu_task *task, struct _starpu_job *j, struct _starpu_worker *worker)
{
	int workerid = worker->workerid;

	int profiling = starpu_profiling_status_get();

	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);

	unsigned index;
	unsigned long total_size = 0;

	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle_t handle = descrs[index].handle;
		enum starpu_data_access_mode mode = descrs[index].mode;
		int node = descrs[index].node;

		if (node < 0)
			continue;

		struct _starpu_data_replicate *local_replicate;
		int needs_init;

		local_replicate = get_replicate(handle, mode, workerid, node);
		_starpu_spin_lock(&handle->header_lock);
		if (local_replicate->mc)
		{
			if (task->prefetched && local_replicate->initialized &&
				/* See prefetch conditions in
				 * starpu_prefetch_task_input_on_node_prio and alike */
				!(mode & (STARPU_SCRATCH|STARPU_REDUX)) &&
				(mode & STARPU_R))
			{
				/* Allocations or transfer prefetches should have been done by now and marked
				 * this mc as needed for us.
				 * Now that we added a reference for the task, we can relieve that.  */
				/* Note: the replicate might have been evicted in between, thus not 100% sure
				 * that our prefetch request is still recorded here.  */
				if (local_replicate->nb_tasks_prefetch > 0)
					local_replicate->nb_tasks_prefetch--;
			}
		}
		if (!(mode & STARPU_R) && (mode & STARPU_W))
		{
			/* The task will be initializing it. Possibly we have
			 * only prefetched the allocation, and now we have to
			 * record that we'll modify it. */
			local_replicate->initialized = 1;
			_starpu_update_data_state(handle, local_replicate, mode);
		}

		needs_init = !local_replicate->initialized;
		_starpu_spin_unlock(&handle->header_lock);

		_STARPU_TASK_SET_INTERFACE(task , local_replicate->data_interface, descrs[index].index);

		/* If the replicate was not initialized yet, we have to do it now */
		if (!(mode & STARPU_SCRATCH) && needs_init)
			_starpu_init_data_replicate(handle, local_replicate, workerid);

#ifdef STARPU_USE_FXT
		if (fut_active)
			total_size += _starpu_data_get_size(handle);
#endif
	}
	_starpu_trace_data_load(workerid,total_size);

	if (profiling && task->profiling_info)
		_starpu_clock_gettime(&task->profiling_info->acquire_data_end_time);

	_starpu_trace_end_fetch_input(NULL);

	_starpu_clear_worker_status(worker, STATUS_INDEX_WAITING, NULL);
}

/* Release task data dependencies */
void __starpu_push_task_output(struct _starpu_job *j)
{
#ifdef STARPU_OPENMP
	STARPU_ASSERT(!j->continuation);
#endif
	int profiling = starpu_profiling_status_get();
	struct starpu_task *task = j->task;
	if (profiling && task->profiling_info)
		_starpu_clock_gettime(&task->profiling_info->release_data_start_time);

	struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);

	int workerid = starpu_worker_get_id();

	unsigned index;
	int indexdup;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle_t handle = descrs[index].handle;
		enum starpu_data_access_mode mode = descrs[index].mode;
		int orig_node = descrs[index].orig_node;
		int node = descrs[index].node;

		struct _starpu_data_replicate *local_replicate = NULL;

		for (indexdup = (int) index-1; indexdup >= 0; indexdup--)
		{
			starpu_data_handle_t handle_dup = descrs[indexdup].handle;
			int node_dup = descrs[indexdup].orig_node;
			if (handle_dup == handle && node_dup == orig_node)
				/* We have already released this data, skip it. This
				 * depends on ordering putting writes before reads, see
				 * _starpu_compar_handles */
				goto next;

			if (!_starpu_handles_same_root(handle_dup, handle))
				/* We are not checking within the same parent any more, no need to continue checking other handles */
				break;
		}

		if (node != -1)
			local_replicate = get_replicate(handle, mode, workerid, node);

		/* Keep a reference for future
		 * _starpu_release_task_enforce_sequential_consistency call */
		_starpu_spin_lock(&handle->header_lock);
		handle->busy_count++;

		if (node == -1)
		{
			/* NOWHERE case, just notify dependencies */
			if (!_starpu_notify_data_dependencies(handle, STARPU_NONE))
				_starpu_spin_unlock(&handle->header_lock);
		}
		else
		{
			_starpu_spin_unlock(&handle->header_lock);
			_starpu_release_data_on_node(handle, 0, STARPU_NONE, local_replicate);
		}
	next:
		;
	}

	if (profiling && task->profiling_info)
		_starpu_clock_gettime(&task->profiling_info->release_data_end_time);
}

/* Version for a driver running on a worker: we show the driver state in the trace */
void _starpu_push_task_output(struct _starpu_job *j)
{
	_starpu_trace_start_push_output(NULL);
	__starpu_push_task_output(j);
	_starpu_trace_end_push_output(NULL);
}

struct fetch_nowhere_wrapper
{
	struct _starpu_job *j;
	unsigned pending;
};

static void _starpu_fetch_nowhere_task_input_cb(void *arg);
/* Asynchronously fetch data for a task which will have no content */
void _starpu_fetch_nowhere_task_input(struct _starpu_job *j)
{
	int profiling = starpu_profiling_status_get();
	struct starpu_task *task = j->task;
	if (profiling && task->profiling_info)
		_starpu_clock_gettime(&task->profiling_info->acquire_data_start_time);

	struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	unsigned nfetchbuffers = 0;
	struct fetch_nowhere_wrapper *wrapper;

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		/* Note here we just follow what was requested, and not use _starpu_task_data_get_node* */
		int node = -1;
		if (task->cl->specific_nodes)
			node = STARPU_CODELET_GET_NODE(task->cl, descrs[index].index);
		descrs[index].node = node;
		if (node != -1)
			nfetchbuffers++;
	}

	if (!nfetchbuffers)
	{
		/* Nothing to fetch actually, already finished! */
		__starpu_push_task_output(j);
		_starpu_handle_job_termination(j);
		_STARPU_LOG_OUT_TAG("handle_job_termination");
		return;
	}

	_STARPU_MALLOC(wrapper, (sizeof(*wrapper)));
	wrapper->j = j;
	/* +1 for the call below */
	wrapper->pending = nfetchbuffers + 1;

	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle_t handle = descrs[index].handle;
		enum starpu_data_access_mode mode = descrs[index].mode;
		int node = descrs[index].node;
		if (node == -1)
			continue;

		if (mode == STARPU_NONE ||
			(mode & ((1<<STARPU_MODE_SHIFT) - 1)) >= STARPU_ACCESS_MODE_MAX ||
			(mode >> STARPU_MODE_SHIFT) >= (STARPU_SHIFTED_MODE_MAX >> STARPU_MODE_SHIFT))
			STARPU_ASSERT_MSG(0, "mode %d (0x%x) is bogus\n", mode, mode);
		STARPU_ASSERT(mode != STARPU_SCRATCH && mode != STARPU_REDUX);

		struct _starpu_data_replicate *local_replicate;

		local_replicate = get_replicate(handle, mode, -1, node);

		_starpu_fetch_data_on_node(handle, node, local_replicate, mode, 0, task, STARPU_FETCH, 1, _starpu_fetch_nowhere_task_input_cb, wrapper, 0, "_starpu_fetch_nowhere_task_input");
	}

	if (profiling && task->profiling_info)
		_starpu_clock_gettime(&task->profiling_info->acquire_data_end_time);

	/* Finished working with the task, release our reference */
	_starpu_fetch_nowhere_task_input_cb(wrapper);
}

static void _starpu_fetch_nowhere_task_input_cb(void *arg)
{
	/* One more transfer finished */
	struct fetch_nowhere_wrapper *wrapper = arg;

	unsigned pending = STARPU_ATOMIC_ADD(&wrapper->pending, -1);
	ANNOTATE_HAPPENS_BEFORE(&wrapper->pending);
	if (pending == 0)
	{
		ANNOTATE_HAPPENS_AFTER(&wrapper->pending);

		/* Finished transferring, task is over */
		struct _starpu_job *j = wrapper->j;
		free(wrapper);
		__starpu_push_task_output(j);
		_starpu_handle_job_termination(j);
		_STARPU_LOG_OUT_TAG("handle_job_termination");
	}
}

/* NB : this value can only be an indication of the status of a data
	at some point, but there is no strong guarantee ! */
unsigned starpu_data_is_on_node(starpu_data_handle_t handle, unsigned node)
{
	unsigned ret = 0;

// XXX : this is just a hint, so we don't take the lock ...
//	STARPU_PTHREAD_SPIN_LOCK(&handle->header_lock);

	if (handle->per_node[node].state != STARPU_INVALID)
	{
		ret  = 1;
	}
	else
	{
		unsigned i;
		unsigned nnodes = starpu_memory_nodes_get_count();

		for (i = 0; i < nnodes; i++)
		{
			if (handle->per_node[node].request[i])
			{
				ret = 1;
				break;
			}
		}

	}

//	STARPU_PTHREAD_SPIN_UNLOCK(&handle->header_lock);

	return ret;
}

/* Return true if a data is on memory and is not part of a prefetch */
unsigned starpu_data_is_on_node_excluding_prefetch(starpu_data_handle_t handle, unsigned node)
{
	return handle->per_node[node].state != STARPU_INVALID;
}

/* Unmap the data from this node, e.g. before partitioning or unregistering */
void _starpu_data_unmap(starpu_data_handle_t handle, unsigned node)
{
	struct _starpu_data_request *r = NULL;
	STARPU_ASSERT(handle);

	_starpu_spin_lock(&handle->header_lock);
	if (handle->per_node[node].mapped != STARPU_UNMAPPED)
	{
		r = _starpu_create_data_request(handle, &handle->per_node[handle->per_node[node].mapped], &handle->per_node[node], node, STARPU_UNMAP, 0, NULL, STARPU_FETCH, 0, 0, __func__);

		r->refcnt++;
		_starpu_post_data_request(r);
	}
	_starpu_spin_unlock(&handle->header_lock);
	if (r)
		_starpu_wait_data_request_completion(r, 1);
}
