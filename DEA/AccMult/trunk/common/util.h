#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <assert.h>
#include <core/jobs.h>

#define PRINTF(str, ...)	fprintf(stderr, str, ## __VA_ARGS__)

#ifndef MIN
#define MIN(a,b)	((a)<(b)?(a):(b))
#endif

#define ASSERT(x)	assert(x)

#define ATOMIC_ADD(ptr, value)  (__sync_fetch_and_add ((ptr), (value)))

void matrix_fill_rand(matrix *);
void matrix_fill_zero(matrix *);
void alloc_matrix(matrix *, unsigned, unsigned);
void free_matrix(matrix *);
void display_matrix(matrix *);
void compare_matrix(matrix *, matrix *, float);

#endif // __UTIL_H__
