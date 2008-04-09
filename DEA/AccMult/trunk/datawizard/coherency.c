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

/* we do have the same code for PPU and SPU in the case of a Cell */
#include "coherency_common.c"
