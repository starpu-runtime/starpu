#ifndef __STARPU_UTIL_H__
#define __STARPU_UTIL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define STARPU_MIN(a,b)	((a)<(b)?(a):(b))
#define STARPU_MAX(a,b)	((a)<(b)?(b):(a))

#define STARPU_ASSERT(x)	assert(x)

#define STARPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#define STARPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))

#define STARPU_ATOMIC_ADD(ptr, value)  (__sync_fetch_and_add ((ptr), (value)) + (value))
#define STARPU_ATOMIC_OR(ptr, value)  (__sync_fetch_and_or ((ptr), (value)))

#define STARPU_SUCCESS	0
#define STARPU_TRYAGAIN	1
#define STARPU_FATAL	2

static int __attribute__ ((unused)) starpu_get_env_number(const char *str)
{
	char *strval;

	strval = getenv(str);
	if (strval) {
		/* the env variable was actually set */
		unsigned val;
		char *check;

		val = (int)strtol(strval, &check, 10);
		STARPU_ASSERT(strcmp(check, "\0") == 0);

		//fprintf(stderr, "ENV %s WAS %d\n", str, val);
		return val;
	}
	else {
		/* there is no such env variable */
		//fprintf("There was no %s ENV\n", str);
		return -1;
	}
}

#endif // __STARPU_UTIL_H__
