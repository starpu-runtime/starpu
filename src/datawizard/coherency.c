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

/* this function will actually copy a valid data into the requesting node */
static int __attribute__((warn_unused_result)) copy_data_to_node(data_state *state, uint32_t requesting_node, 
						 unsigned donotread)
{
	/* first find a valid copy, either a OWNER or a SHARED */
	int ret;
	uint32_t node;
	uint32_t src_node_mask = 0;
	for (node = 0; node < MAXNODES; node++)
	{
		if (state->per_node[node].state != INVALID) {
			/* we found a copy ! */
			src_node_mask |= (1<<node);
		}
	}

	/* we should have found at least one copy ! */
	STARPU_ASSERT(src_node_mask != 0);

	ret = driver_copy_data(state, src_node_mask, requesting_node, donotread);

	return ret;
}

/* this may be called once the data is fetched with header and STARPU_RW-lock hold */
static void update_data_state(data_state *state, uint32_t requesting_node,
				uint8_t write)
{
	/* the data is present now */
	state->per_node[requesting_node].requested = 0;

	if (write) {
		/* the requesting node now has the only valid copy */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			state->per_node[node].state = INVALID;
		}
		state->per_node[requesting_node].state = OWNER;
	}
	else { /* read only */
		/* there was at least another copy of the data */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			if (state->per_node[node].state != INVALID)
				state->per_node[node].state = SHARED;
		}
		state->per_node[requesting_node].state = SHARED;
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

int _fetch_data(data_state *state, uint32_t requesting_node,
			uint8_t read, uint8_t write)
{
	while (take_mutex_try(&state->header_lock)) {
		datawizard_progress(requesting_node);
	}

	cache_state local_state;
	local_state = state->per_node[requesting_node].state;

	/* we handle that case first to optimize the OWNER path */
	if ((local_state == OWNER) || (local_state == SHARED && !write))
	{
		/* the local node already got its data */
		release_mutex(&state->header_lock);
		msi_cache_hit(requesting_node);
		return 0;
	}

	if ((local_state == SHARED) && write) {
		/* local node already has the data but it must invalidate 
		 * other copies */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			if (state->per_node[node].state == SHARED) 
			{
				state->per_node[node].state =
					(node == requesting_node ? OWNER:INVALID);
			}

		}
		
		release_mutex(&state->header_lock);
		msi_cache_hit(requesting_node);
		return 0;
	}

	/* the only remaining situation is that the local copy was invalid */
	STARPU_ASSERT(state->per_node[requesting_node].state == INVALID);

	msi_cache_miss(requesting_node);

	/* we need the data from either the owner or one of the sharer */
	int ret;
	ret = copy_data_to_node(state, requesting_node, !read);
	if (ret != 0)
	switch (ret) {
		case -ENOMEM:
			goto enomem;
		
		default:
			STARPU_ASSERT(0);
	}

	update_data_state(state, requesting_node, write);

	release_mutex(&state->header_lock);

	return 0;

enomem:
	/* there was not enough local memory to fetch the data */
	release_mutex(&state->header_lock);
	return -ENOMEM;
}

static int fetch_data(data_state *state, starpu_access_mode mode)
{
	int ret;
	uint32_t requesting_node = get_local_memory_node(); 

	uint8_t read, write;
	read = (mode != STARPU_W); /* then R or STARPU_RW */
	write = (mode != STARPU_R); /* then STARPU_W or STARPU_RW */

#ifndef NO_DATA_RW_LOCK
	if (write) {
//		take_rw_lock_write(&state->data_lock);
		while (take_rw_lock_write_try(&state->data_lock))
			datawizard_progress(requesting_node);
	} else {
//		take_rw_lock_read(&state->data_lock);
		while (take_rw_lock_read_try(&state->data_lock))
			datawizard_progress(requesting_node);
	}
#endif

	while (take_mutex_try(&state->header_lock))
		datawizard_progress(requesting_node);

	state->per_node[requesting_node].refcnt++;
	release_mutex(&state->header_lock);

	ret = _fetch_data(state, requesting_node, read, write);
	if (ret != 0)
		goto enomem;

	return 0;
enomem:
	/* we did not get the data so remove the lock anyway */
	while (take_mutex_try(&state->header_lock))
		datawizard_progress(requesting_node);

	state->per_node[requesting_node].refcnt--;
	release_mutex(&state->header_lock);

#ifndef NO_DATA_RW_LOCK
	release_rw_lock(&state->data_lock);
#endif

	return -1;
}

uint32_t get_data_refcnt(data_state *state, uint32_t node)
{
	return state->per_node[node].refcnt;
}

/* in case the data was accessed on a write mode, do not forget to 
 * make it accessible again once it is possible ! */
static void release_data(data_state *state, uint32_t default_wb_mask)
{
	uint32_t wb_mask;

	/* normally, the requesting node should have the data in an exclusive manner */
	uint32_t requesting_node = get_local_memory_node();
	STARPU_ASSERT(state->per_node[requesting_node].state != INVALID);

	wb_mask = default_wb_mask | state->wb_mask;

	/* are we doing write-through or just some normal write-back ? */
	if (wb_mask & ~(1<<requesting_node)) {
		write_through_data(state, requesting_node, wb_mask);
	}

	while (take_mutex_try(&state->header_lock))
		datawizard_progress(requesting_node);

	state->per_node[requesting_node].refcnt--;
	release_mutex(&state->header_lock);

#ifndef NO_DATA_RW_LOCK
	/* this is intended to make data accessible again */
	release_rw_lock(&state->data_lock);
#else
	notify_data_dependencies(state);
#endif
}

int fetch_codelet_input(starpu_buffer_descr *descrs, starpu_data_interface_t *interface, unsigned nbuffers, uint32_t mask)
{
	TRACE_START_FETCH_INPUT(NULL);

	uint32_t local_memory_node = get_local_memory_node();

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		int ret;
		starpu_buffer_descr *descr;
		data_state *state;

		descr = &descrs[index];

		state = descr->state;

		ret = fetch_data(state, descr->mode);
		if (STARPU_UNLIKELY(ret))
			goto enomem;

		memcpy(&interface[index], &state->interface[local_memory_node], 
				sizeof(starpu_data_interface_t));
	}

	TRACE_END_FETCH_INPUT(NULL);

	return 0;

enomem:
	/* try to unreference all the input that were successfully taken */
	fprintf(stderr, "something went wrong with buffer %d\n", index);
	push_codelet_output(descrs, index, mask);
	return -1;
}

void push_codelet_output(starpu_buffer_descr *descrs, unsigned nbuffers, uint32_t mask)
{
	TRACE_START_PUSH_OUTPUT(NULL);

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		release_data(descrs[index].state, mask);
	}

	TRACE_END_PUSH_OUTPUT(NULL);
}

int request_data_allocation(data_state *state, uint32_t node)
{
	take_mutex(&state->header_lock);

	int ret;
	ret = allocate_per_node_buffer(state, node);
	STARPU_ASSERT(ret == 0);

	/* XXX quick and dirty hack */
	state->per_node[node].automatically_allocated = 0;	

	release_mutex(&state->header_lock);

	return 0;
}

#ifdef NO_DATA_RW_LOCK
struct state_and_node {
	data_state *state;
	unsigned node;
	pthread_cond_t cond;
	pthread_mutex_t lock;
	unsigned finished;
};
#endif

#ifdef NO_DATA_RW_LOCK
/* put the current value of the data into RAM */
static inline void _starpu_sync_data_with_mem_continuation(void *arg)
{
	int ret;
	struct state_and_node *statenode = arg;

	data_state *state = statenode->state;

	ret = fetch_data(state, STARPU_R);
	
	STARPU_ASSERT(!ret);
	
	/* the application does not need to "lock" the data anymore */
	notify_data_dependencies(state);

	pthread_mutex_lock(&statenode->lock);
	statenode->finished = 1;
	pthread_cond_signal(&statenode->cond);
	pthread_mutex_unlock(&statenode->lock);
}
#endif

void starpu_sync_data_with_mem(data_state *state)
{
	int ret;

#ifdef NO_DATA_RW_LOCK
	struct state_and_node statenode =
	{
		.state = state,
		.node = 0, /* unused here */
		.cond = PTHREAD_COND_INITIALIZER,
		.lock = PTHREAD_MUTEX_INITIALIZER,
		.finished = 0
	};

	/* we try to get the data, if we do not succeed immediately, we set a
 	* callback function that will be executed automatically when the data is
 	* available again, otherwise we fetch the data directly */
	if (!attempt_to_submit_data_request_from_apps(state, STARPU_R, 
			_starpu_sync_data_with_mem_continuation, &statenode))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_sync_data_with_mem_continuation(&statenode);
	}
	else {
		pthread_mutex_lock(&statenode.lock);
		if (!statenode.finished)
			pthread_cond_wait(&statenode.cond, &statenode.lock);
		pthread_mutex_unlock(&statenode.lock);
	}
#else
	/* NB: fetch_data automatically grabs the STARPU_RW-lock so it needs to be
 	 * released explicitely afterward */
	ret = fetch_data(state, STARPU_R);
	STARPU_ASSERT(!ret);

	release_rw_lock(&state->data_lock);
#endif
}

static inline void do_notify_data_modification(data_state *state, uint32_t modifying_node)
{
	take_mutex(&state->header_lock);

	unsigned node = 0;
	for (node = 0; node < MAXNODES; node++)
	{
		state->per_node[node].state =
			(node == modifying_node?OWNER:INVALID);
	}

	release_mutex(&state->header_lock);
}

#ifdef NO_DATA_RW_LOCK
static inline void _notify_data_modification_continuation(void *arg)
{
	struct state_and_node *statenode = arg;

	do_notify_data_modification(statenode->state, statenode->node);

	pthread_mutex_lock(&statenode->lock);
	statenode->finished = 1;
	pthread_cond_signal(&statenode->cond);
	pthread_mutex_unlock(&statenode->lock);
}
#endif

/* in case the application did modify the data ... invalidate all other copies  */
void starpu_notify_data_modification(data_state *state, uint32_t modifying_node)
{
	/* this may block .. XXX */
#ifdef NO_DATA_RW_LOCK
	struct state_and_node statenode =
	{
		.state = state,
		.node = modifying_node,
		.cond = PTHREAD_COND_INITIALIZER,
		.lock = PTHREAD_MUTEX_INITIALIZER,
		.finished = 0
	};

	if (!attempt_to_submit_data_request_from_apps(state, STARPU_W, _notify_data_modification_continuation, &statenode))
	{
		/* we can immediately proceed */
		do_notify_data_modification(state, modifying_node);
	}
	else {
		pthread_mutex_lock(&statenode.lock);
		if (!statenode.finished)
			pthread_cond_wait(&statenode.cond, &statenode.lock);
		pthread_mutex_unlock(&statenode.lock);
	}

	/* remove the "lock"/reference */
	notify_data_dependencies(state);

#else
	take_rw_lock_write(&state->data_lock);

	do_notify_data_modification(state, modifying_node);

	release_rw_lock(&state->data_lock);
#endif
}

/* NB : this value can only be an indication of the status of a data
	at some point, but there is no strong garantee ! */
unsigned is_data_present_or_requested(data_state *state, uint32_t node)
{
	unsigned ret = 0;

// XXX : this is just a hint, so we don't take the lock ...
//	take_mutex(&state->header_lock);

	if (state->per_node[node].state != INVALID 
		|| state->per_node[node].requested)
		ret = 1;

//	release_mutex(&state->header_lock);

	return ret;
}

inline void set_data_requested_flag_if_needed(data_state *state, uint32_t node)
{
// XXX : this is just a hint, so we don't take the lock ...
//	take_mutex(&state->header_lock);

	if (state->per_node[node].state == INVALID) 
		state->per_node[node].requested = 1;

//	release_mutex(&state->header_lock);
}
