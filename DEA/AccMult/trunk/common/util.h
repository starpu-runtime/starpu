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

#define ATOMIC_OR(ptr, value)  (__sync_fetch_and_or ((ptr), (value)))

static int get_env_number(const char *str)
{
	char *strval;

	strval = getenv(str);
	if (strval) {
		/* the env variable was actually set */
		unsigned val;
		char *check;

		val = (int)strtol(strval, &check, 10);
		ASSERT(strcmp(check, "\0") == 0);

		//fprintf(stderr, "ENV %s WAS %d\n", str, val);
		return val;
	}
	else {
		/* there is no such env variable */
		//fprintf("There was no %s ENV\n", str);
		return -1;
	}
}

#endif // __UTIL_H__
