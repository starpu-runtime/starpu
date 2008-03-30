#include "coherency.h"

void take_lock(data_lock *lock)
{
	uint32_t prev;
	do {
		prev = __sync_lock_test_and_set(&lock->taken, TAKEN);
	} while (prev == TAKEN);
}

void release_lock(data_lock *lock)
{
	lock->taken = FREE;
}

void monitor_new_data(data_state *state, uint32_t home_node, 
			uintptr_t ptr, size_t length)
{
	ASSERT(state);

	/* initialize the new lock */
	state->lock.taken = FREE;

	/* first take care to properly lock the data */
	take_lock(&state->lock);

	/* we assume that all nodes may use that data */
	state->nnodes = MAXNODES;

	/* make sure we do have a valid copy */
	ASSERT(home_node < MAXNODES);

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (node == home_node) {
			/* this is the home node with the only valid copy */
			state->per_node[node].state = OWNER;
			state->per_node[node].ptr = ptr;
			state->per_node[node].allocated = 1;
			state->per_node[node].automatically_allocated = 0;
		}
		else {
			/* the value is not available here yet */
			state->per_node[node].state = INVALID;
			state->per_node[node].ptr = 0;
			state->per_node[node].allocated = 0;
		}
	}

	state->length = length;

	/* now the data is available ! */
	release_lock(&state->lock);
}

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
void copy_data_to_node(data_state *state, uint32_t requesting_node)
{
	/* first find a valid copy, either a OWNER or a SHARED */
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
	ASSERT(src_node_mask != 0);

	driver_copy_data(state, src_node_mask, requesting_node);
	return;
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
 * 	data copy + invalid->shared + owner->shared (ASSERT(there is a valid))
 * case 3 : invalid + write : 
 * 	no data copy + invalid->owner + (owner,shared)->invalid
 * case 4 : invalid + R/W : 
 * 	data copy + if (W) (invalid->owner + owner->invalid) else (invalid,owner->shared)
 */
uintptr_t fetch_data(data_state *state, uint32_t requesting_node,
			uint8_t read, uint8_t write)
{
	take_lock(&state->lock);

//	printf("FETCH from %d R,W = %d,%d\n", requesting_node, read, write);

	cache_state local_state;
	local_state = state->per_node[requesting_node].state;

	/* we handle that case first to optimize the OWNER path */
	if ((local_state == OWNER) || (local_state == SHARED && !write))
	{
		/* the local node already got its data */

		if (!write) {
			/* else, do not forget to call release_data !*/
			release_lock(&state->lock);
		}

		return state->per_node[requesting_node].ptr;
	}

	if ((local_state == SHARED) && write) {
		/* local node already has the data but it must invalidate other copies */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			if (state->per_node[node].state == SHARED) 
			{
				state->per_node[node].state =
					(node == requesting_node ? OWNER:INVALID);
			}

		}
		
		/* do not forget to use a release_data */
		return state->per_node[requesting_node].ptr;
	}

	/* the only remaining situation is that the local copy was invalid */
	ASSERT(state->per_node[requesting_node].state == INVALID);

	/* we first need to copy the data from either the owner or one of the sharer */
	copy_data_to_node(state, requesting_node);

	if (write) {
		/* the requesting node now has the only valid copy */
		uint32_t node;
		for (node = 0; node < MAXNODES; node++)
		{
			state->per_node[node].state = INVALID;
		}
		state->per_node[requesting_node].state = OWNER;

		/* do not forget to release the data later on ! */
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
		release_lock(&state->lock);
	}

	return state->per_node[requesting_node].ptr;
}

void write_through_data(data_state *state, uint32_t requesting_node, uint32_t write_through_mask)
{
	/* first commit all changes onto the nodes specified by the mask */
	uint32_t node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (write_through_mask & (1<<node)) {
			/* we need to commit the buffer on that node */
			if (node != requesting_node) 
			{
//				printf("write_through_data %d -> %d \n", requesting_node, node);
				/* the requesting node already has the data by definition */
				driver_copy_data_1_to_1(state, requesting_node, node);
			}
				
			/* now the data is shared among the nodes on the write_through_mask */
			state->per_node[node].state = SHARED;
		}
	}

	/* the requesting node is now one sharer */
	if (write_through_mask & ~(1<<requesting_node))
	{
		state->per_node[requesting_node].state = SHARED;
	}
}

/* in case the data was accessed on a write mode, do not forget to 
 * make it accessible again once it is possible ! */
void release_data(data_state *state, uint32_t requesting_node, uint32_t write_through_mask)
{
	/* normally, the requesting node should have the data in an exclusive manner */
	ASSERT(state->per_node[requesting_node].state == OWNER);
	
	/* are we doing write-through or just some normal write-back ? */
	if (write_through_mask & ~(1<<requesting_node))
		write_through_data(state, requesting_node, write_through_mask);

	/* this is intended to make data accessible again */
	release_lock(&state->lock);
}

