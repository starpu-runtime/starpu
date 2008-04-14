#ifndef __RWLOCKS_H__
#define __RWLOCKS_H__

#include <stdint.h>
#include "util.h"

typedef struct rw_lock_t {
	uint8_t busy;
	uint8_t writer;
	uint16_t readercnt;
} rw_lock;

void init_rw_lock(rw_lock *lock);
void take_rw_lock_write(rw_lock *lock);
void take_rw_lock_read(rw_lock *lock);
void release_rw_lock(rw_lock *lock);

#endif
