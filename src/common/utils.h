/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

/** @file */

#include <common/config.h>
#include <starpu.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#ifdef STARPU_HAVE_SCHED_YIELD
#include <sched.h>
#endif

#ifdef STARPU_HAVE_HELGRIND_H
#include <valgrind/helgrind.h>
#endif

#ifndef DO_CREQ_v_WW
#define DO_CREQ_v_WW(_creqF, _ty1F, _arg1F, _ty2F, _arg2F) ((void)0)
#endif
#ifndef DO_CREQ_v_W
#define DO_CREQ_v_W(_creqF, _ty1F, _arg1F) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_BEFORE
#define ANNOTATE_HAPPENS_BEFORE(obj) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_BEFORE_FORGET_ALL
#define ANNOTATE_HAPPENS_BEFORE_FORGET_ALL(obj) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_AFTER
#define ANNOTATE_HAPPENS_AFTER(obj) ((void)0)
#endif
#ifndef VALGRIND_HG_DISABLE_CHECKING
#define VALGRIND_HG_DISABLE_CHECKING(start, len) ((void)0)
#endif
#ifndef VALGRIND_HG_ENABLE_CHECKING
#define VALGRIND_HG_ENABLE_CHECKING(start, len) ((void)0)
#endif
#ifndef VALGRIND_STACK_REGISTER
#define VALGRIND_STACK_REGISTER(stackbottom, stacktop) 0
#endif
#ifndef VALGRIND_STACK_DEREGISTER
#define VALGRIND_STACK_DEREGISTER(id) ((void)0)
#endif
#ifndef RUNNING_ON_VALGRIND
#define RUNNING_ON_VALGRIND 0
#endif
#ifdef STARPU_SANITIZE_THREAD
#define STARPU_RUNNING_ON_VALGRIND 1
#else
#define STARPU_RUNNING_ON_VALGRIND RUNNING_ON_VALGRIND
#endif
#define STARPU_HG_DISABLE_CHECKING(variable) VALGRIND_HG_DISABLE_CHECKING(&(variable), sizeof(variable))
#define STARPU_HG_ENABLE_CHECKING(variable)  VALGRIND_HG_ENABLE_CHECKING(&(variable), sizeof(variable))

#if defined(__KNC__) || defined(__KNF__)
#define STARPU_DEBUG_PREFIX "[starpu-mic]"
#else
#define STARPU_DEBUG_PREFIX "[starpu]"
#endif

/* This is needed in some places to make valgrind yield to another thread to be
 * able to progress.  */
#if defined(__i386__) || defined(__x86_64__)
#define _STARPU_UYIELD() __asm__ __volatile("rep; nop")
#else
#define _STARPU_UYIELD() ((void)0)
#endif
#if defined(STARPU_HAVE_SCHED_YIELD) && defined(STARPU_HAVE_HELGRIND_H)
#define STARPU_VALGRIND_YIELD() do { if (STARPU_RUNNING_ON_VALGRIND) sched_yield(); } while (0)
#define STARPU_UYIELD() do { if (STARPU_RUNNING_ON_VALGRIND) sched_yield(); else _STARPU_UYIELD(); } while (0)
#else
#define STARPU_VALGRIND_YIELD() do { } while (0)
#define STARPU_UYIELD() _STARPU_UYIELD()
#endif

#ifdef STARPU_VERBOSE
#  define _STARPU_DEBUG(fmt, ...) do { if (!_starpu_silent) {fprintf(stderr, STARPU_DEBUG_PREFIX"[%s] " fmt ,__starpu_func__ ,## __VA_ARGS__); fflush(stderr); }} while(0)
#  define _STARPU_DEBUG_NO_HEADER(fmt, ...) do { if (!_starpu_silent) {fprintf(stderr, fmt , ## __VA_ARGS__); fflush(stderr); }} while(0)
#else
#  define _STARPU_DEBUG(fmt, ...) do { } while (0)
#  define _STARPU_DEBUG_NO_HEADER(fmt, ...) do { } while (0)
#endif

#ifdef STARPU_EXTRA_VERBOSE
#  define _STARPU_EXTRA_DEBUG(fmt, ...) do { if (!_starpu_silent) {fprintf(stderr, STARPU_DEBUG_PREFIX"[%s] " fmt ,__starpu_func__ ,## __VA_ARGS__); fflush(stderr); }} while(0)
#else
#  define _STARPU_EXTRA_DEBUG(fmt, ...) do { } while (0)
#endif

#ifdef STARPU_EXTRA_VERBOSE
#  define _STARPU_LOG_IN()             do { if (!_starpu_silent) {fprintf(stderr, STARPU_DEBUG_PREFIX"[%ld][%s:%s@%d] -->\n", starpu_pthread_self(), __starpu_func__,__FILE__,  __LINE__); }} while(0)
#  define _STARPU_LOG_OUT()            do { if (!_starpu_silent) {fprintf(stderr, STARPU_DEBUG_PREFIX"[%ld][%s:%s@%d] <--\n", starpu_pthread_self(), __starpu_func__, __FILE__,  __LINE__); }} while(0)
#  define _STARPU_LOG_OUT_TAG(outtag)  do { if (!_starpu_silent) {fprintf(stderr, STARPU_DEBUG_PREFIX"[%ld][%s:%s@%d] <-- (%s)\n", starpu_pthread_self(), __starpu_func__, __FILE__, __LINE__, outtag); }} while(0)
#else
#  define _STARPU_LOG_IN()
#  define _STARPU_LOG_OUT()
#  define _STARPU_LOG_OUT_TAG(outtag)
#endif

/* TODO: cache */
#define _STARPU_MSG(fmt, ...) do { fprintf(stderr, STARPU_DEBUG_PREFIX"[%s] " fmt ,__starpu_func__ ,## __VA_ARGS__); } while(0)
#define _STARPU_DISP(fmt, ...) do { if (!_starpu_silent) {fprintf(stderr, STARPU_DEBUG_PREFIX"[%s] " fmt ,__starpu_func__ ,## __VA_ARGS__); }} while(0)
#define _STARPU_ERROR(fmt, ...)                                                  \
	do {                                                                          \
                fprintf(stderr, "\n\n[starpu][%s] Error: " fmt ,__starpu_func__ ,## __VA_ARGS__);    \
		fprintf(stderr, "\n\n");					      \
		STARPU_ABORT();                                                       \
	} while (0)


#ifdef _MSC_VER
#  if defined(__cplusplus)
#    define _STARPU_DECLTYPE(x) (decltype(x))
#  else
#    define _STARPU_DECLTYPE(x)
#  endif
#else
#  define _STARPU_DECLTYPE(x) (__typeof(x))
#endif

#define _STARPU_MALLOC(ptr, size) do { ptr = _STARPU_DECLTYPE(ptr) malloc(size); STARPU_ASSERT_MSG(ptr != NULL || size == 0, "Cannot allocate %ld bytes\n", (long) (size)); } while (0)
#define _STARPU_CALLOC(ptr, nmemb, size) do { ptr = _STARPU_DECLTYPE(ptr) calloc(nmemb, size); STARPU_ASSERT_MSG(ptr != NULL || size == 0, "Cannot allocate %ld bytes\n", (long) (nmemb*size)); } while (0)
#define _STARPU_REALLOC(ptr, size) do { void *_new_ptr = realloc(ptr, size); STARPU_ASSERT_MSG(_new_ptr != NULL || size == 0, "Cannot reallocate %ld bytes\n", (long) (size)); ptr = _STARPU_DECLTYPE(ptr) _new_ptr;} while (0)

#ifdef _MSC_VER
#define _STARPU_IS_ZERO(a) (a == 0.0)
#else
#define _STARPU_IS_ZERO(a) (fpclassify(a) == FP_ZERO)
#endif

char *_starpu_mkdtemp_internal(char *tmpl);
char *_starpu_mkdtemp(char *tmpl);
int _starpu_mkpath(const char *s, mode_t mode);
void _starpu_mkpath_and_check(const char *s, mode_t mode);
char *_starpu_mktemp(const char *directory, int flags, int *fd);
/** This version creates a hierarchy of n temporary directories, useful when
 * creating a lot of temporary files to be stored in the same place */
char *_starpu_mktemp_many(const char *directory, int depth, int flags, int *fd);
void _starpu_rmtemp_many(char *path, int depth);
void _starpu_rmdir_many(char *path, int depth);
int _starpu_fftruncate(FILE *file, size_t length);
int _starpu_ftruncate(int fd, size_t length);
int _starpu_frdlock(FILE *file);
int _starpu_frdunlock(FILE *file);
int _starpu_fwrlock(FILE *file);
int _starpu_fwrunlock(FILE *file);
char *_starpu_get_home_path(void);
void _starpu_gethostname(char *hostname, size_t size);

/** If FILE is currently on a comment line, eat it.  */
void _starpu_drop_comments(FILE *f);

struct _starpu_job;
/** Returns the symbol associated to that job if any. */
const char *_starpu_job_get_model_name(struct _starpu_job *j);
/** Returns the name associated to that job if any. */
const char *_starpu_job_get_task_name(struct _starpu_job *j);

struct starpu_codelet;
/** Returns the symbol associated to that job if any. */
const char *_starpu_codelet_get_model_name(struct starpu_codelet *cl);

int _starpu_check_mutex_deadlock(starpu_pthread_mutex_t *mutex);

void _starpu_util_init(void);

#endif // __COMMON_UTILS_H__
