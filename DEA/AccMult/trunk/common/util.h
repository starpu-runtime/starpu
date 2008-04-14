#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <assert.h>

#define PRINTF(str, ...)	fprintf(stderr, str, ## __VA_ARGS__)

#ifndef MIN
#define MIN(a,b)	((a)<(b)?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b)	((a)<(b)?(b):(a))
#endif

#ifndef ASSERT
#define ASSERT(x)	assert(x)
#endif

#define UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#define LIKELY(expr)            (__builtin_expect(!!(expr),1))

#define ATOMIC_ADD(ptr, value)  (__sync_fetch_and_add ((ptr), (value)) + (value))

#endif // __UTIL_H__
