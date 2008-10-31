#include "mutex.h"

void init_mutex(mutex *m)
{
	/* this is free at first */
	m->taken = 0;
}

void take_mutex(mutex *m)
{
	uint32_t prev;
	do {
		prev = __sync_lock_test_and_set(&m->taken, 1);
	} while (prev);
}

void release_mutex(mutex *m)
{
	m->taken = 0;
}
