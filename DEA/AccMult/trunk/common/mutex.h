#ifndef __MUTEX_H__
#define __MUTEX_H__

#include <stdint.h>

typedef struct mutex_t {
	/* we only have a trivial implementation yet ! */
	volatile uint32_t taken __attribute__ ((aligned(16)));
#ifdef USE_SPU
	uintptr_t ea_taken;
#endif
} mutex;

void init_mutex(mutex *m);
void take_mutex(mutex *m);
void release_mutex(mutex *m);

#endif
