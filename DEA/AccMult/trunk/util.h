#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <assert.h>

#define PRINTF(str, ...)	fprintf(stderr, str, ## __VA_ARGS__)

#ifndef MIN
#define MIN(a,b)	((a)<(b)?(a):(b))
#endif

#define ASSERT(x)	assert(x)

#define ATOMIC_ADD(ptr, value)  (__sync_fetch_and_add ((ptr), (value)))

#endif // __UTIL_H__
