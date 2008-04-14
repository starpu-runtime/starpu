/**
 * A dummy implementation of a rw_lock using spinlocks ...
 */ 

#include "rwlock.h"

static void _take_busy_lock(rw_lock *lock)
{
	uint8_t prev;
	do {
		prev = __sync_lock_test_and_set(&lock->busy, 1);
	} while (prev);
}

static void _release_busy_lock(rw_lock *lock)
{
	lock->busy = 0;
}

void init_rw_lock(rw_lock *lock)
{
	ASSERT(lock);

	lock->writer = 0;
	lock->readercnt = 0;
	lock->busy = 0;
}

void take_rw_lock_write(rw_lock *lock)
{
	do {
		_take_busy_lock(lock);
		
		if (lock->readercnt > 0 || lock->writer)
		{
			/* fail to take the lock */
			_release_busy_lock(lock);
		}
		else {
			ASSERT(lock->readercnt == 0);
			ASSERT(lock->writer == 0);

			/* no one was either writing nor reading */
			lock->writer = 1;
			_release_busy_lock(lock);
			return;
		}
	} while (1);
}

void take_rw_lock_read(rw_lock *lock)
{
	do {
		if (lock->writer)
		{
			/* there is a writer ... */
			_release_busy_lock(lock);
		}
		else {
			ASSERT(lock->writer == 0);

			/* no one is writing */
			/* XXX check wrap arounds ... */
			lock->readcnt++;
			_release_busy_lock(lock);
			return;
		}
	} while (1);
}

void release_rw_lock(rw_lock *lock)
{
	_take_busy_lock(lock);
	/* either writer or reader (exactly one !) */
	if (lock->writer) 
	{
		ASSERT(lock->readcnt == 0);
		lock->writer = 0;
	}
	else {
		/* reading mode */
		ASSERT(lock->writer == 0);
		lock->readcnt--;
	}
	_release_busy_lock(lock);
}

