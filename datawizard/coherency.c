#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <core/dependencies/data-concurrency.h>

void display_state(data_state *state)
{
	uint32_t node;

	printf("******************************************\n");
	for (node = 0; node < MAXNODES; node++)
	{
		switch(state->per_node[node].state) {
			case INVALID:
				printf("\t%d\tINVALID\n", node);
				break;
			case OWNER:
				printf("\t%d\tOWNER\n", node);
				break;
			case SHARED:
				printf("\t%d\tSHARED\n", node);
				break;
		}
	}

	printf("******************************************\n");

}

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

/* this may be called once the data is fetched with header and RW-lock hold */
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
 *			R 	W 	RW
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
 * case 4 : invalid + R/W : 
 * 	data copy + if (W) (invalid->owner + owner->invalid) 
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

int fetch_data(data_state *state, access_mode mode)
{
	int ret;
	uint32_t requesting_node = get_local_memory_node(); 

	uint8_t read, write;
	read = (mode != W); /* then R or RW */
	write = (mode != R); /* then W or RW */

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

void write_through_data(data_state *state, uint32_t requesting_node, 
					   uint32_t write_through_mask)
{
	if ((write_through_mask & ~(1<<requesting_node)) == 0) {
		/* nothing will be done ... */
		return;
	}

	while (take_mutex_try(&state->header_lock))
		datawizard_progress(requesting_node);

	/* first commit all changes onto the nodes specified by the mask */
	uint32_t node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (write_through_mask & (1<<node)) {
			/* we need to commit the buffer on that node */
			if (node != requesting_node) 
			{
				/* the requesting node already has the data by
				 * definition */
				int ret;
				ret = driver_copy_data_1_to_1(state, 
						requesting_node, node, 0);

				/* there must remain memory on the write-through mask to honor the request */
				if (ret)
					STARPU_ASSERT(0);
			}
				
			/* now the data is shared among the nodes on the
			 * write_through_mask */
			state->per_node[node].state = SHARED;
		}
	}

	/* the requesting node is now one sharer */
	if (write_through_mask & ~(1<<requesting_node))
	{
		state->per_node[requesting_node].state = SHARED;
	}

	release_mutex(&state->header_lock);
}

/* in case the data was accessed on a write mode, do not forget to 
 * make it accessible again once it is possible ! */
static void release_data(data_state *state, uint32_t write_through_mask)
{
	/* normally, the requesting node should have the data in an exclusive manner */
	uint32_t requesting_node = get_local_memory_node();
	STARPU_ASSERT(state->per_node[requesting_node].state != INVALID);

	/* are we doing write-through or just some normal write-back ? */
	if (write_through_mask & ~(1<<requesting_node)) {
		write_through_data(state, requesting_node, write_through_mask);
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

int fetch_codelet_input(buffer_descr *descrs, data_interface_t *interface, unsigned nbuffers, uint32_t mask)
{
	TRACE_START_FETCH_INPUT(NULL);

	uint32_t local_memory_node = get_local_memory_node();

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		int ret;
		buffer_descr *descr;
		data_state *state;

		descr = &descrs[index];

		state = descr->state;

		ret = fetch_data(state, descr->mode);
		if (UNLIKELY(ret))
			goto enomem;

		descr->interfaceid = state->interfaceid;

		memcpy(&interface[index], &state->interface[local_memory_node], 
				sizeof(data_interface_t));
	}

	TRACE_END_FETCH_INPUT(NULL);

	return 0;

enomem:
	/* try to unreference all the input that were successfully taken */
	printf("something went wrong with buffer %d\n", index);
	push_codelet_output(descrs, index, mask);
	return -1;
}

void push_codelet_output(buffer_descr *descrs, unsigned nbuffers, uint32_t mask)
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

/* in case the application did modify the data ... invalidate all other copies  */
void notify_data_modification(data_state *state, uint32_t modifying_node)
{
	/* this may block .. XXX */
#ifndef NO_DATA_RW_LOCK
	take_rw_lock_write(&state->data_lock);
#else
#warning notify_data_modification is not supported with NO_DATA_RW_LOCK yet
#endif

	take_mutex(&state->header_lock);

	unsigned node = 0;
	for (node = 0; node < MAXNODES; node++)
	{
		state->per_node[node].state =
			(node == modifying_node?OWNER:INVALID);
	}

	release_mutex(&state->header_lock);
#ifndef NO_DATA_RW_LOCK
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
