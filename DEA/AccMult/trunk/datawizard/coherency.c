#include "coherency.h"

extern void driver_copy_data(data_state *state, uint32_t src_node_mask, uint32_t dst_node);
extern void driver_copy_data_1_to_1(data_state *state, uint32_t node, uint32_t requesting_node);

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
#ifdef USE_SPU
	state->ea_data_state = (uintptr_t)&state;
	state->lock.ea_taken = (uintptr_t)&state->lock.taken;
#endif

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

/* we do have the same code for PPU and SPU in the case of a Cell */
#include "coherency_common.c"
