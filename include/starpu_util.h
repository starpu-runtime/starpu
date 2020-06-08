/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <starpu_config.h>

#ifdef __GLIBC__
#include <execinfo.h>
#endif

#ifdef STARPU_SIMGRID_MC
#include <simgrid/modelchecker.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Toolbox Toolbox
   @brief The following macros allow to make GCC extensions portable,
   and to have a code which can be compiled with any C compiler.
   @{
*/

/**
   Return true (non-zero) if GCC version \p maj.\p min or later is
   being used (macro taken from glibc.)
*/
#if defined __GNUC__ && defined __GNUC_MINOR__
# define STARPU_GNUC_PREREQ(maj, min) \
	((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
# define STARPU_GNUC_PREREQ(maj, min) 0
#endif

/**
   When building with a GNU C Compiler, allow programmers to mark an
   expression as unlikely.
*/
#ifdef __GNUC__
#  define STARPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#else
#  define STARPU_UNLIKELY(expr)          (expr)
#endif

/**
   When building with a GNU C Compiler, allow programmers to mark an
   expression as likely.
*/
#ifdef __GNUC__
#  define STARPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))
#else
#  define STARPU_LIKELY(expr)            (expr)
#endif

/**
   When building with a GNU C Compiler, defined to __attribute__((unused))
*/
#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_UNUSED                  __attribute__((unused))
#else
#  define STARPU_ATTRIBUTE_UNUSED
#endif

/**
   When building with a GNU C Compiler, defined to __attribute__((noreturn))
*/
#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_NORETURN                  __attribute__((noreturn))
#else
#  define STARPU_ATTRIBUTE_NORETURN
#endif

/**
   When building with a GNU C Compiler, defined to __attribute__((visibility ("internal")))
*/
#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_INTERNAL      __attribute__ ((visibility ("internal")))
#else
#  define STARPU_ATTRIBUTE_INTERNAL
#endif

/**
   When building with a GNU C Compiler, defined to __attribute__((malloc))
*/
#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_MALLOC                  __attribute__((malloc))
#else
#  define STARPU_ATTRIBUTE_MALLOC
#endif

/**
   When building with a GNU C Compiler, defined to __attribute__((warn_unused_result))
*/
#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_WARN_UNUSED_RESULT      __attribute__((warn_unused_result))
#else
#  define STARPU_ATTRIBUTE_WARN_UNUSED_RESULT
#endif

/**
   When building with a GNU C Compiler, defined to  __attribute__((pure))
*/
#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_PURE                    __attribute__((pure))
#else
#  define STARPU_ATTRIBUTE_PURE
#endif

/**
   When building with a GNU C Compiler, defined to__attribute__((aligned(size)))
*/
#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_ALIGNED(size)           __attribute__((aligned(size)))
#else
#  define STARPU_ATTRIBUTE_ALIGNED(size)
#endif

#ifdef __GNUC__
#  define STARPU_ATTRIBUTE_FORMAT(type, string, first)                  __attribute__((format(type, string, first)))
#else
#  define STARPU_ATTRIBUTE_FORMAT(type, string, first)
#endif

/* Note that if we're compiling C++, then just use the "inline"
   keyword, since it's part of C++ */
#if defined(c_plusplus) || defined(__cplusplus)
#  define STARPU_INLINE inline
#elif defined(_MSC_VER) || defined(__HP_cc)
#  define STARPU_INLINE __inline
#else
#  define STARPU_INLINE __inline__
#endif

#if STARPU_GNUC_PREREQ(4, 3)
#  define STARPU_ATTRIBUTE_CALLOC_SIZE(num,size)   __attribute__((alloc_size(num,size)))
#  define STARPU_ATTRIBUTE_ALLOC_SIZE(size)        __attribute__((alloc_size(size)))
#else
#  define STARPU_ATTRIBUTE_CALLOC_SIZE(num,size)
#  define STARPU_ATTRIBUTE_ALLOC_SIZE(size)
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

#define STARPU_BACKTRACE_LENGTH	32
#ifdef __GLIBC__
#  define STARPU_DUMP_BACKTRACE() do { \
	void *__ptrs[STARPU_BACKTRACE_LENGTH]; \
	int __n = backtrace(__ptrs, STARPU_BACKTRACE_LENGTH); \
	backtrace_symbols_fd(__ptrs, __n, 2); \
} while (0)
#else
#  define STARPU_DUMP_BACKTRACE() do { } while (0)
#endif

#ifdef STARPU_SIMGRID_MC
#define STARPU_SIMGRID_ASSERT(x) MC_assert(!!(x))
#else
#define STARPU_SIMGRID_ASSERT(x)
#endif

/**
   Unless StarPU has been configured with the option \ref enable-fast
   "--enable-fast", this macro will abort if the expression \p x is false.
*/
#ifdef STARPU_NO_ASSERT
#define STARPU_ASSERT(x)		do { if (0) { (void) (x); } } while(0)
#else
#  if defined(__CUDACC__) || defined(STARPU_HAVE_WINDOWS)
#    define STARPU_ASSERT(x)		do { if (STARPU_UNLIKELY(!(x))) { STARPU_DUMP_BACKTRACE(); STARPU_SIMGRID_ASSERT(x); *(int*)NULL = 0; } } while(0)
#  else
#    define STARPU_ASSERT(x)		do { if (STARPU_UNLIKELY(!(x))) { STARPU_DUMP_BACKTRACE(); STARPU_SIMGRID_ASSERT(x); assert(x); } } while (0)
#  endif
#endif

#ifdef STARPU_NO_ASSERT
#define STARPU_ASSERT_ACCESSIBLE(x)	do { if (0) { (void) (x); } } while(0)
#else
#define STARPU_ASSERT_ACCESSIBLE(ptr)	do { volatile char __c STARPU_ATTRIBUTE_UNUSED = *(char*) (ptr); } while(0)
#endif

/**
   Unless StarPU has been configured with the option \ref enable-fast
   "--enable-fast", this macro will abort if the expression \p x is false.
   The string \p msg will be displayed.
*/
#ifdef STARPU_NO_ASSERT
#define STARPU_ASSERT_MSG(x, msg, ...)	do { if (0) { (void) (x); (void) msg; } } while(0)
#else
#  if defined(__CUDACC__) || defined(STARPU_HAVE_WINDOWS)
#    define STARPU_ASSERT_MSG(x, msg, ...)	do { if (STARPU_UNLIKELY(!(x))) { STARPU_DUMP_BACKTRACE(); fprintf(stderr, "\n[starpu][%s][assert failure] " msg "\n\n", __starpu_func__, ## __VA_ARGS__); STARPU_SIMGRID_ASSERT(x); *(int*)NULL = 0; }} while(0)
#  else
#    define STARPU_ASSERT_MSG(x, msg, ...)	do { if (STARPU_UNLIKELY(!(x))) { STARPU_DUMP_BACKTRACE(); fprintf(stderr, "\n[starpu][%s][assert failure] " msg "\n\n", __starpu_func__, ## __VA_ARGS__); STARPU_SIMGRID_ASSERT(x); assert(x); } } while(0)
#  endif
#endif

#ifdef __APPLE_CC__
#  ifdef __clang_analyzer__
#    define _starpu_abort() exit(42)
#  else
#    define _starpu_abort() *(volatile int*)NULL = 0
#  endif
#else
#  define _starpu_abort() abort()
#endif

/**
   Abort the program.
*/
#define STARPU_ABORT() do {                                          \
	STARPU_DUMP_BACKTRACE();                                     \
        fprintf(stderr, "[starpu][abort][%s()@%s:%d]\n", __starpu_func__, __FILE__, __LINE__); \
	_starpu_abort();				\
} while(0)

/**
   Print the string '[starpu][abort][name of the calling function:name
   of the file:line in the file]' followed by the given string \p msg
   and abort the program
*/
#define STARPU_ABORT_MSG(msg, ...) do {					\
	STARPU_DUMP_BACKTRACE();                                        \
	fprintf(stderr, "[starpu][abort][%s()@%s:%d] " msg "\n", __starpu_func__, __FILE__, __LINE__, ## __VA_ARGS__); \
	_starpu_abort();				\
} while(0)

#if defined(STARPU_HAVE_STRERROR_R)
#if (! defined(__GLIBC__) || !__GLIBC__) || ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && (! defined(_GNU_SOURCE)))
/* XSI-compliant version of strerror_r returns an int */
#       define starpu_strerror_r(errnum, buf, buflen) \
	do \
	{ \
		int _ret = strerror_r((errnum), (buf), (buflen)); \
		STARPU_ASSERT(_ret == 0); \
	} \
	while (0)
#else
/* GNU-specific version of strerror_r returns a char * */
#       define starpu_strerror_r(errnum, buf, buflen) \
	do \
	{ \
		char * const _user_buf = (buf); \
		const size_t _user_buflen = (buflen); \
		/* the GNU-specific behaviour when 'buf' == NULL cannot be emulated with the XSI-compliant version */ \
		STARPU_ASSERT((buf) != NULL); \
		char * _tmp_buf = strerror_r((errnum), _user_buf, _user_buflen); \
		if (_tmp_buf != _user_buf) \
		{ \
			if (_user_buflen > 0) \
			{ \
				strncpy(_user_buf, _tmp_buf, _user_buflen-1); \
				_user_buf[_user_buflen-1] = '\0'; \
			} \
		} \
	} \
	while (0)
#endif /* strerror_r ABI version */
#endif  /* STARPU_HAVE_STRERROR_R */

/**
   Abort the program (after displaying \p message) if \p err has a
   value which is not 0.
*/
#if defined(STARPU_HAVE_STRERROR_R)
#  define STARPU_CHECK_RETURN_VALUE(err, message, ...) {if (STARPU_UNLIKELY(err != 0)) { \
			char xmessage[256]; starpu_strerror_r(-err, xmessage, 256); \
			fprintf(stderr, "[starpu] Unexpected value: <%d:%s> returned for " message "\n", err, xmessage, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#else
#  define STARPU_CHECK_RETURN_VALUE(err, message, ...) {if (STARPU_UNLIKELY(err != 0)) { \
			fprintf(stderr, "[starpu] Unexpected value: <%d> returned for " message "\n", err, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#endif

/**
   Abort the program (after displaying \p message) if \p err is
   different from \p value.
*/
#if defined(STARPU_HAVE_STRERROR_R)
#  define STARPU_CHECK_RETURN_VALUE_IS(err, value, message, ...) {if (STARPU_UNLIKELY(err != value)) { \
			char xmessage[256]; starpu_strerror_r(-err, xmessage, 256); \
			fprintf(stderr, "[starpu] Unexpected value: <%d!=%d:%s> returned for " message "\n", err, value, xmessage, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#else
#  define STARPU_CHECK_RETURN_VALUE_IS(err, value, message, ...) {if (STARPU_UNLIKELY(err != value)) { \
	       		fprintf(stderr, "[starpu] Unexpected value: <%d != %d> returned for " message "\n", err, value, ## __VA_ARGS__); \
			STARPU_ABORT(); }}
#endif

/* Note: do not use _starpu_cmpxchg / _starpu_xchg / _starpu_cmpxchgl /
 * _starpu_xchgl / _starpu_cmpxchg64 / _starpu_xchg64, which only
 * assembly-hand-written fallbacks used when building with an old gcc.
 * Rather use STARPU_VAL_COMPARE_AND_SWAP and STARPU_VAL_EXCHANGE available on
 * all platforms with a recent-enough gcc */

#if defined(__i386__) || defined(__x86_64__)
static __starpu_inline unsigned _starpu_cmpxchg(unsigned *ptr, unsigned old, unsigned next)
{
	__asm__ __volatile__("lock cmpxchgl %2,%1": "+a" (old), "+m" (*ptr) : "q" (next) : "memory");
	return old;
}
#define STARPU_HAVE_CMPXCHG
static __starpu_inline unsigned _starpu_xchg(unsigned *ptr, unsigned next)
{
	/* Note: xchg is always locked already */
	__asm__ __volatile__("xchgl %1,%0": "+m" (*ptr), "+q" (next) : : "memory");
	return next;
}
#define STARPU_HAVE_XCHG

static __starpu_inline uint32_t _starpu_cmpxchg32(uint32_t *ptr, uint32_t old, uint32_t next)
{
	__asm__ __volatile__("lock cmpxchgl %2,%1": "+a" (old), "+m" (*ptr) : "q" (next) : "memory");
	return old;
}
#define STARPU_HAVE_CMPXCHG32
static __starpu_inline uint32_t _starpu_xchg32(uint32_t *ptr, uint32_t next)
{
	/* Note: xchg is always locked already */
	__asm__ __volatile__("xchgl %1,%0": "+m" (*ptr), "+q" (next) : : "memory");
	return next;
}
#define STARPU_HAVE_XCHG32

#if defined(__i386__)
static __starpu_inline unsigned long _starpu_cmpxchgl(unsigned long *ptr, unsigned long old, unsigned long next)
{
	__asm__ __volatile__("lock cmpxchgl %2,%1": "+a" (old), "+m" (*ptr) : "q" (next) : "memory");
	return old;
}
#define STARPU_HAVE_CMPXCHGL
static __starpu_inline unsigned long _starpu_xchgl(unsigned long *ptr, unsigned long next)
{
	/* Note: xchg is always locked already */
	__asm__ __volatile__("xchgl %1,%0": "+m" (*ptr), "+q" (next) : : "memory");
	return next;
}
#define STARPU_HAVE_XCHGL
#endif

#if defined(__x86_64__)
static __starpu_inline unsigned long _starpu_cmpxchgl(unsigned long *ptr, unsigned long old, unsigned long next)
{
	__asm__ __volatile__("lock cmpxchgq %2,%1": "+a" (old), "+m" (*ptr) : "q" (next) : "memory");
	return old;
}
#define STARPU_HAVE_CMPXCHGL
static __starpu_inline unsigned long _starpu_xchgl(unsigned long *ptr, unsigned long next)
{
	/* Note: xchg is always locked already */
	__asm__ __volatile__("xchgq %1,%0": "+m" (*ptr), "+q" (next) : : "memory");
	return next;
}
#define STARPU_HAVE_XCHGL
#endif

#if defined(__i386__)
static __starpu_inline uint64_t _starpu_cmpxchg64(uint64_t *ptr, uint64_t old, uint64_t next)
{
	uint32_t next_hi = next >> 32;
	uint32_t next_lo = next & 0xfffffffful;
	__asm__ __volatile__("lock cmpxchg8b %1": "+A" (old), "+m" (*ptr) : "c" (next_hi), "b" (next_lo) : "memory");
	return old;
}
#define STARPU_HAVE_CMPXCHG64
#endif

#if defined(__x86_64__)
static __starpu_inline uint64_t _starpu_cmpxchg64(uint64_t *ptr, uint64_t old, uint64_t next)
{
	__asm__ __volatile__("lock cmpxchgq %2,%1": "+a" (old), "+m" (*ptr) : "q" (next) : "memory");
	return old;
}
#define STARPU_HAVE_CMPXCHG64
static __starpu_inline uint64_t _starpu_xchg64(uint64_t *ptr, uint64_t next)
{
	/* Note: xchg is always locked already */
	__asm__ __volatile__("xchgq %1,%0": "+m" (*ptr), "+q" (next) : : "memory");
	return next;
}
#define STARPU_HAVE_XCHG64
#endif

#endif

#define STARPU_ATOMIC_SOMETHING(name,expr) \
static __starpu_inline unsigned starpu_atomic_##name(unsigned *ptr, unsigned value) \
{ \
	unsigned old, next; \
	while (1) \
	{ \
		old = *ptr; \
		next = expr; \
		if (_starpu_cmpxchg(ptr, old, next) == old) \
			break; \
	}; \
	return expr; \
}
#define STARPU_ATOMIC_SOMETHINGL(name,expr) \
static __starpu_inline unsigned long starpu_atomic_##name##l(unsigned long *ptr, unsigned long value) \
{ \
	unsigned long old, next; \
	while (1) \
	{ \
		old = *ptr; \
		next = expr; \
		if (_starpu_cmpxchgl(ptr, old, next) == old) \
			break; \
	}; \
	return expr; \
}
#define STARPU_ATOMIC_SOMETHING64(name,expr) \
static __starpu_inline uint64_t starpu_atomic_##name##64(uint64_t *ptr, uint64_t value) \
{ \
	uint64_t old, next; \
	while (1) \
	{ \
		old = *ptr; \
		next = expr; \
		if (_starpu_cmpxchg64(ptr, old, next) == old) \
			break; \
	}; \
	return expr; \
}

/* Returns the new value */
#ifdef STARPU_HAVE_SYNC_FETCH_AND_ADD
#define STARPU_ATOMIC_ADD(ptr, value)  (__sync_fetch_and_add ((ptr), (value)) + (value))
#define STARPU_ATOMIC_ADDL(ptr, value)  (__sync_fetch_and_add ((ptr), (value)) + (value))
#define STARPU_ATOMIC_ADD64(ptr, value)  (__sync_fetch_and_add ((ptr), (value)) + (value))
#else
#if defined(STARPU_HAVE_CMPXCHG)
STARPU_ATOMIC_SOMETHING(add, old + value)
#define STARPU_ATOMIC_ADD(ptr, value) starpu_atomic_add(ptr, value)
#endif
#if defined(STARPU_HAVE_CMPXCHGL)
STARPU_ATOMIC_SOMETHINGL(add, old + value)
#define STARPU_ATOMIC_ADDL(ptr, value) starpu_atomic_addl(ptr, value)
#endif
#if defined(STARPU_HAVE_CMPXCHG64)
STARPU_ATOMIC_SOMETHING64(add, old + value)
#define STARPU_ATOMIC_ADD64(ptr, value) starpu_atomic_add64(ptr, value)
#endif
#endif

#ifdef STARPU_HAVE_SYNC_FETCH_AND_OR
#define STARPU_ATOMIC_OR(ptr, value)  (__sync_fetch_and_or ((ptr), (value)))
#define STARPU_ATOMIC_ORL(ptr, value)  (__sync_fetch_and_or ((ptr), (value)))
#define STARPU_ATOMIC_OR64(ptr, value)  (__sync_fetch_and_or ((ptr), (value)))
#else
#if defined(STARPU_HAVE_CMPXCHG)
STARPU_ATOMIC_SOMETHING(or, old | value)
#define STARPU_ATOMIC_OR(ptr, value) starpu_atomic_or(ptr, value)
#endif
#if defined(STARPU_HAVE_CMPXCHGL)
STARPU_ATOMIC_SOMETHINGL(or, old | value)
#define STARPU_ATOMIC_ORL(ptr, value) starpu_atomic_orl(ptr, value)
#endif
#if defined(STARPU_HAVE_CMPXCHG64)
STARPU_ATOMIC_SOMETHING64(or, old | value)
#define STARPU_ATOMIC_OR64(ptr, value) starpu_atomic_or64(ptr, value)
#endif
#endif

#ifdef STARPU_HAVE_SYNC_BOOL_COMPARE_AND_SWAP
#define STARPU_BOOL_COMPARE_AND_SWAP(ptr, old, value)  (__sync_bool_compare_and_swap ((ptr), (old), (value)))
#define STARPU_BOOL_COMPARE_AND_SWAP32(ptr, old, value) STARPU_BOOL_COMPARE_AND_SWAP(ptr, old, value)
#define STARPU_BOOL_COMPARE_AND_SWAP64(ptr, old, value) STARPU_BOOL_COMPARE_AND_SWAP(ptr, old, value)
#else
#ifdef STARPU_HAVE_CMPXCHG
#define STARPU_BOOL_COMPARE_AND_SWAP(ptr, old, value) (_starpu_cmpxchg((ptr), (old), (value)) == (old))
#endif
#ifdef STARPU_HAVE_CMPXCHG32
#define STARPU_BOOL_COMPARE_AND_SWAP32(ptr, old, value) (_starpu_cmpxchg32((ptr), (old), (value)) == (old))
#endif
#ifdef STARPU_HAVE_CMPXCHG64
#define STARPU_BOOL_COMPARE_AND_SWAP64(ptr, old, value) (_starpu_cmpxchg64((ptr), (old), (value)) == (old))
#endif
#endif

#ifdef STARPU_HAVE_SYNC_VAL_COMPARE_AND_SWAP
#define STARPU_VAL_COMPARE_AND_SWAP(ptr, old, value)  (__sync_val_compare_and_swap ((ptr), (old), (value)))
#define STARPU_VAL_COMPARE_AND_SWAP32(ptr, old, value) STARPU_VAL_COMPARE_AND_SWAP(ptr, old, value)
#define STARPU_VAL_COMPARE_AND_SWAP64(ptr, old, value) STARPU_VAL_COMPARE_AND_SWAP(ptr, old, value)
#else
#ifdef STARPU_HAVE_CMPXCHG
#define STARPU_VAL_COMPARE_AND_SWAP(ptr, old, value) (_starpu_cmpxchg((ptr), (old), (value)))
#endif
#ifdef STARPU_HAVE_CMPXCHG32
#define STARPU_VAL_COMPARE_AND_SWAP32(ptr, old, value) (_starpu_cmpxchg32((ptr), (old), (value)))
#endif
#ifdef STARPU_HAVE_CMPXCHG64
#define STARPU_VAL_COMPARE_AND_SWAP64(ptr, old, value) (_starpu_cmpxchg64((ptr), (old), (value)))
#endif
#endif

#ifdef STARPU_HAVE_ATOMIC_EXCHANGE_N
#define STARPU_VAL_EXCHANGE(ptr, value) (__atomic_exchange_n((ptr), (value), __ATOMIC_SEQ_CST))
#define STARPU_VAL_EXCHANGEL(ptr, value) STARPU_VAL_EXCHANGE((ptr) (value))
#define STARPU_VAL_EXCHANGE32(ptr, value) STARPU_VAL_EXCHANGE((ptr) (value))
#define STARPU_VAL_EXCHANGE64(ptr, value) STARPU_VAL_EXCHANGE((ptr) (value))
#else
#ifdef STARPU_HAVE_XCHG
#define STARPU_VAL_EXCHANGE(ptr, value) (_starpu_xchg((ptr), (value)))
#endif
#ifdef STARPU_HAVE_XCHGL
#define STARPU_VAL_EXCHANGEL(ptr, value) (_starpu_xchgl((ptr), (value)))
#endif
#ifdef STARPU_HAVE_XCHG32
#define STARPU_VAL_EXCHANGE32(ptr, value) (_starpu_xchg32((ptr), (value)))
#endif
#ifdef STARPU_HAVE_XCHG64
#define STARPU_VAL_EXCHANGE64(ptr, value) (_starpu_xchg64((ptr), (value)))
#endif
#endif

/* Returns the previous value */
#ifdef STARPU_HAVE_SYNC_LOCK_TEST_AND_SET
#define STARPU_TEST_AND_SET(ptr, value) (__sync_lock_test_and_set ((ptr), (value)))
#define STARPU_RELEASE(ptr) (__sync_lock_release ((ptr)))
#elif defined(STARPU_HAVE_XCHG)
#define STARPU_TEST_AND_SET(ptr, value) (_starpu_xchg((ptr), (value)))
#define STARPU_RELEASE(ptr) (_starpu_xchg((ptr), 0))
#endif

#ifdef STARPU_HAVE_SYNC_SYNCHRONIZE
#define STARPU_SYNCHRONIZE() __sync_synchronize()
#elif defined(__i386__)
#define STARPU_SYNCHRONIZE() __asm__ __volatile__("lock; addl $0,0(%%esp)" ::: "memory")
#elif defined(__KNC__) || defined(__KNF__)
#define STARPU_SYNCHRONIZE() __asm__ __volatile__("lock; addl $0,0(%%rsp)" ::: "memory")
#elif defined(__x86_64__)
#define STARPU_SYNCHRONIZE() __asm__ __volatile__("mfence" ::: "memory")
#elif defined(__ppc__) || defined(__ppc64__)
#define STARPU_SYNCHRONIZE() __asm__ __volatile__("sync" ::: "memory")
#endif

/**
   This macro can be used to do a synchronization.
*/
#if defined(__i386__)
#define STARPU_RMB() __asm__ __volatile__("lock; addl $0,0(%%esp)" ::: "memory")
#elif defined(__KNC__) || defined(__KNF__)
#define STARPU_RMB() __asm__ __volatile__("lock; addl $0,0(%%rsp)" ::: "memory")
#elif defined(__x86_64__)
#define STARPU_RMB() __asm__ __volatile__("lfence" ::: "memory")
#elif defined(__ppc__) || defined(__ppc64__)
#define STARPU_RMB() __asm__ __volatile__("sync" ::: "memory")
#else
#define STARPU_RMB() STARPU_SYNCHRONIZE()
#endif

/**
   This macro can be used to do a synchronization.
*/
#if defined(__i386__)
#define STARPU_WMB() __asm__ __volatile__("lock; addl $0,0(%%esp)" ::: "memory")
#elif defined(__KNC__) || defined(__KNF__)
#define STARPU_WMB() __asm__ __volatile__("lock; addl $0,0(%%rsp)" ::: "memory")
#elif defined(__x86_64__)
#define STARPU_WMB() __asm__ __volatile__("sfence" ::: "memory")
#elif defined(__ppc__) || defined(__ppc64__)
#define STARPU_WMB() __asm__ __volatile__("sync" ::: "memory")
#else
#define STARPU_WMB() STARPU_SYNCHRONIZE()
#endif

#if defined(__i386__) || defined(__x86_64__)
#define STARPU_CACHELINE_SIZE 64
#elif defined(__ppc__) || defined(__ppc64__) || defined(__ia64__)
#define STARPU_CACHELINE_SIZE 128
#elif defined(__s390__) || defined(__s390x__)
#define STARPU_CACHELINE_SIZE 256
#else
/* Conservative default */
#define STARPU_CACHELINE_SIZE 1024
#endif

#ifdef _WIN32
/* Try to fetch the system definition of timespec */
#include <sys/types.h>
#include <sys/stat.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <time.h>
#if !defined(_MSC_VER) || defined(BUILDING_STARPU)
#include <pthread.h>
#endif
#if !defined(STARPU_HAVE_STRUCT_TIMESPEC) || (defined(_MSC_VER) && _MSC_VER < 1900)
/* If it didn't get defined in the standard places, then define it ourself */
#ifndef STARPU_TIMESPEC_DEFINED
#define STARPU_TIMESPEC_DEFINED 1
struct timespec
{
     time_t  tv_sec;  /* Seconds */
     long    tv_nsec; /* Nanoseconds */
};
#endif /* STARPU_TIMESPEC_DEFINED */
#endif /* STARPU_HAVE_STRUCT_TIMESPEC */
/* Fetch gettimeofday on mingw/cygwin */
#if defined(__MINGW32__) || defined(__CYGWIN__)
#include <sys/time.h>
#endif
#else
#include <sys/time.h>
#endif /* _WIN32 */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_UTIL_H__ */
