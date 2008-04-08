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

/* we do have the same code for PPU and SPU in the case of a Cell */
#include "coherency_common.c"
