#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>

#define PRINTF(str, ...)	fprintf(stderr, str, ## __VA_ARGS__)

#define MIN(a,b)	((a)<(b)?(a):(b))

#endif // __UTIL_H__
