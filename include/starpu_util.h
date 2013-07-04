/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_UTIL_H__
#define __STARPU_UTIL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <starpu_config.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* Return true (non-zero) if GCC version MAJ.MIN or later is being used
 * (macro taken from glibc.)  */
#if defined __GNUC__ && defined __GNUC_MINOR__
# define STARPU_GNUC_PREREQ(maj, min) \
	((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
# define STARPU_GNUC_PREREQ(maj, min) 0
#endif

#ifdef __GNUC__
#  define STARPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#  define STARPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))
#  define STARPU_ATTRIBUTE_UNUSED                  __attribute__((unused))
#  define STARPU_ATTRIBUTE_INTERNAL      __attribute__ ((visibility ("internal")))
#  define STARPU_ATTRIBUTE_MALLOC                  __attribute__((malloc))
#  define STARPU_ATTRIBUTE_WARN_UNUSED_RESULT      __attribute__((warn_unused_result))
#  define STARPU_ATTRIBUTE_PURE                    __attribute__((pure))
#  define STARPU_ATTRIBUTE_ALIGNED(size)           __attribute__((aligned(size)))
#else
#  define STARPU_UNLIKELY(expr)          (expr)
#  define STARPU_LIKELY(expr)            (expr)
#  define STARPU_ATTRIBUTE_UNUSED
#  define STARPU_ATTRIBUTE_INTERNAL
#  define STARPU_ATTRIBUTE_MALLOC
#  define STARPU_ATTRIBUTE_WARN_UNUSED_RESULT
#  define STARPU_ATTRIBUTE_PURE
#  define STARPU_ATTRIBUTE_ALIGNED(size)
#endif

#if STARPU_GNUC_PREREQ(3, 1) && !defined(BUILDING_STARPU) && !defined(STARPU_USE_DEPRECATED_API) && !defined(STARPU_USE_DEPRECATED_ONE_ZERO_API)
#define STARPU_DEPRECATED  __attribute__((__deprecated__))
#else
#define STARPU_DEPRECATED
#endif /* __GNUC__ */

#if STARPU_GNUC_PREREQ(3,3)
#define STARPU_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#define STARPU_WARN_UNUSED_RESULT
#endif /* __GNUC__ */

#define STARPU_POISON_PTR	((void *)0xdeadbeef)

#define STARPU_MIN(a,b)	((a)<(b)?(a):(b))
#define STARPU_MAX(a,b)	((a)<(b)?(b):(a))

#ifdef STARPU_NO_ASSERT
#define STARPU_ASSERT(x)		do { } while(0)
#define STARPU_ASSERT_MSG(x, msg, ...)	do { } while(0)
#else
#  if defined(__CUDACC__) && defined(STARPU_HAVE_WINDOWS)
#    define STARPU_ASSERT(x)		do { if (STARPU_UNLIKELY(!(x))) *(int*)NULL = 0; } while(0)
#    define STARPU_ASSERT_MSG(x, msg, ...)	do { if (STARPU_UNLIKELY(!(x))) { fprintf(stderr, "\n[starpu][%s][assert failure] " msg "\n", __starpu_func__, ## __VA_ARGS__); *(int*)NULL = 0; }} while(0)
#  else
#    define STARPU_ASSERT(x)		assert(x)
#    define STARPU_ASSERT_MSG(x, msg, ...)	do { if (STARPU_UNLIKELY(!(x))) { fprintf(stderr, "\n[starpu][%s][assert failure] " msg "\n", __starpu_func__, ## __VA_ARGS__); } ; assert(x); } while(0)

#  endif
#endif

#define STARPU_ABORT() do {                                          \
	fprintf(stderr, "[starpu][abort][%s@%s:%d]\n", __starpu_func__, __FILE__, __LINE__); \
	abort();                                                     \
} while(0)

#define STARPU_ABORT_MSG(msg, ...) do {					\
	fprintf(stderr, "[starpu][abort][%s@%s:%d] " msg "\n", __starpu_func__, __FILE__, __LINE__, ## __VA_ARGS__); \
	abort();                                                     \
} while(0)

#if defined(STARPU_HAVE_STRERROR_R)
#  define STARPU_CHECK_RETURN_VALUE(err, message, ...) {if (STARPU_UNLIKELY(err != 0)) { \
			char xmessage[256]; strerror_r(-err, xmessage, 256); \
			fprintf(stderr, "[starpu] Unexpected value: <%d:%s> returned for " message "\n", err, xmessage, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#  define STARPU_CHECK_RETURN_VALUE_IS(err, value, message, ...) {if (STARPU_UNLIKELY(err != value)) { \
			char xmessage[256]; strerror_r(-err, xmessage, 256); \
			fprintf(stderr, "[starpu] Unexpected value: <%d!=%d:%s> returned for " message "\n", err, value, xmessage, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#else
#  define STARPU_CHECK_RETURN_VALUE(err, message, ...) {if (STARPU_UNLIKELY(err != 0)) { \
			fprintf(stderr, "[starpu] Unexpected value: <%d> returned for " message "\n", err, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#  define STARPU_CHECK_RETURN_VALUE_IS(err, value, message, ...) {if (STARPU_UNLIKELY(err != value)) { \
	       		fprintf(stderr, "[starpu] Unexpected value: <%d != %d> returned for " message "\n", err, value, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#endif /* STARPU_HAVE_STRERROR_R */

#if defined(__i386__) || defined(__x86_64__)

static __starpu_inline unsigned starpu_cmpxchg(unsigned *ptr, unsigned old, unsigned next)
{
	__asm__ __volatile__("lock cmpxchgl %2,%1": "+a" (old), "+m" (*ptr) : "q" (next) : "memory");
	return old;
}
static __starpu_inline unsigned starpu_xchg(unsigned *ptr, unsigned next)
{
	/* Note: xchg is always locked already */
	__asm__ __volatile__("xchgl %1,%0": "+m" (*ptr), "+q" (next) : : "memory");
	return next;
}
#define STARPU_HAVE_XCHG
#endif

#define STARPU_ATOMIC_SOMETHING(name,expr) \
static __starpu_inline unsigned starpu_atomic_##name(unsigned *ptr, unsigned value) \
{ \
	unsigned old, next; \
	while (1) \
	{ \
		old = *ptr; \
		next = expr; \
		if (starpu_cmpxchg(ptr, old, next) == old) \
			break; \
	}; \
	return expr; \
}

#ifdef STARPU_HAVE_SYNC_FETCH_AND_ADD
#define STARPU_ATOMIC_ADD(ptr, value)  (__sync_fetch_and_add ((ptr), (value)) + (value))
#elif defined(STARPU_HAVE_XCHG)
STARPU_ATOMIC_SOMETHING(add, old + value)
#define STARPU_ATOMIC_ADD(ptr, value) starpu_atomic_add(ptr, value)
#endif

#ifdef STARPU_HAVE_SYNC_FETCH_AND_OR
#define STARPU_ATOMIC_OR(ptr, value)  (__sync_fetch_and_or ((ptr), (value)))
#elif defined(STARPU_HAVE_XCHG)
STARPU_ATOMIC_SOMETHING(or, old | value)
#define STARPU_ATOMIC_OR(ptr, value) starpu_atomic_or(ptr, value)
#endif

#ifdef STARPU_HAVE_SYNC_BOOL_COMPARE_AND_SWAP
#define STARPU_BOOL_COMPARE_AND_SWAP(ptr, old, value)  (__sync_bool_compare_and_swap ((ptr), (old), (value)))
#elif defined(STARPU_HAVE_XCHG)
#define STARPU_BOOL_COMPARE_AND_SWAP(ptr, old, value) (starpu_cmpxchg((ptr), (old), (value)) == (old))
#endif

#ifdef STARPU_HAVE_SYNC_LOCK_TEST_AND_SET
#define STARPU_TEST_AND_SET(ptr, value) (__sync_lock_test_and_set ((ptr), (value)))
#define STARPU_RELEASE(ptr) (__sync_lock_release ((ptr)))
#elif defined(STARPU_HAVE_XCHG)
#define STARPU_TEST_AND_SET(ptr, value) (starpu_xchg((ptr), (value)))
#define STARPU_RELEASE(ptr) (starpu_xchg((ptr), 0))
#endif

#ifdef STARPU_HAVE_SYNC_SYNCHRONIZE
#define STARPU_SYNCHRONIZE() __sync_synchronize()
#elif defined(__i386__)
#define STARPU_SYNCHRONIZE() __asm__ __volatile__("lock; addl $0,0(%%esp)" ::: "memory")
#elif defined(__x86_64__)
#define STARPU_SYNCHRONIZE() __asm__ __volatile__("mfence" ::: "memory")
#elif defined(__ppc__) || defined(__ppc64__)
#define STARPU_SYNCHRONIZE() __asm__ __volatile__("sync" ::: "memory")
#endif

#if defined(__i386__) || defined(__KNC__) || defined(__KNF__)
#define STARPU_RMB() __asm__ __volatile__("lock; addl $0,0(%%esp)" ::: "memory")
#define STARPU_WMB() __asm__ __volatile__("lock; addl $0,0(%%esp)" ::: "memory")
#elif defined(__x86_64__)
#define STARPU_RMB() __asm__ __volatile__("lfence" ::: "memory")
#define STARPU_WMB() __asm__ __volatile__("sfence" ::: "memory")
#elif defined(__ppc__) || defined(__ppc64__)
#define STARPU_RMB() __asm__ __volatile__("sync" ::: "memory")
#define STARPU_WMB() __asm__ __volatile__("sync" ::: "memory")
#else
#define STARPU_RMB() STARPU_SYNCHRONIZE()
#define STARPU_WMB() STARPU_SYNCHRONIZE()
#endif

/* This is needed in some places to make valgrind yield to another thread to be
 * able to progress.  */
#if defined(__i386__) || defined(__x86_64__)
#define STARPU_UYIELD() __asm__ __volatile("rep; nop")
#else
#define STARPU_UYIELD() ((void)0)
#endif

#ifdef __cplusplus
}
#endif

/* Include this only here so that <starpu_data_interfaces.h> can use the
 * macros above.  */
#include <starpu_task.h>

#ifdef __cplusplus
extern "C"
{
#endif

static __starpu_inline int starpu_get_env_number(const char *str)
{
	char *strval;

	strval = getenv(str);
	if (strval)
	{
		/* the env variable was actually set */
		long int val;
		char *check;

		val = strtol(strval, &check, 10);
		if (*check) {
			fprintf(stderr,"The %s environment variable must contain an integer\n", str);
			STARPU_ABORT();
		}

		/* fprintf(stderr, "ENV %s WAS %d\n", str, val); */
		return (int)val;
	}
	else
	{
		/* there is no such env variable */
		/* fprintf("There was no %s ENV\n", str); */
		return -1;
	}
}

/* Add an event in the execution trace if FxT is enabled */
void starpu_trace_user_event(unsigned long code);

void starpu_execute_on_each_worker(void (*func)(void *), void *arg, uint32_t where);

/* Same as starpu_execute_on_each_worker, except that the task name is specified in the "name" parameter. */
void starpu_execute_on_each_worker_ex(void (*func)(void *), void *arg, uint32_t where, const char * name);

/* Call func(arg) on every worker in the "workers" array. "num_workers"
 * indicates the number of workers in this array.  This function is
 * synchronous, but the different workers may execute the function in parallel.
 * */
void starpu_execute_on_specific_workers(void (*func)(void*), void * arg, unsigned num_workers, unsigned * workers, const char * name);

int starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle, int asynchronous, void (*callback_func)(void*), void *callback_arg);

/* Return the current date in us */
double starpu_timing_now(void);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_UTIL_H__ */
