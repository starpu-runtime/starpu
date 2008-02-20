#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <assert.h>

#define PRINTF(str, ...)	fprintf(stderr, str, ## __VA_ARGS__)

#ifndef MIN
#define MIN(a,b)	((a)<(b)?(a):(b))
#endif

#define ASSERT(x)	assert(x)

#endif // __UTIL_H__
