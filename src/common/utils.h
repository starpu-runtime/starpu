/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2013  Université de Bordeaux 1
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

#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

#include <starpu.h>
#include <common/config.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
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
#ifndef ANNOTATE_HAPPENS_AFTER
#define ANNOTATE_HAPPENS_AFTER(obj) ((void)0)
#endif
#ifndef VALGRIND_HG_DISABLE_CHECKING
#define VALGRIND_HG_DISABLE_CHECKING(start, len) ((void)0)
#endif
#ifndef VALGRIND_HG_ENABLE_CHECKING
#define VALGRIND_HG_ENABLE_CHECKING(start, len) ((void)0)
#endif
#define STARPU_HG_DISABLE_CHECKING(variable) VALGRIND_HG_DISABLE_CHECKING(&(variable), sizeof(variable))
#define STARPU_HG_ENABLE_CHECKING(variable)  VALGRIND_HG_ENABLE_CHECKING(&(variable), sizeof(variable))

/* This is needed in some places to make valgrind yield to another thread to be
 * able to progress.  */
#if defined(__i386__) || defined(__x86_64__)
#define _STARPU_UYIELD() __asm__ __volatile("rep; nop")
#else
#define _STARPU_UYIELD() ((void)0)
#endif
#if defined(STARPU_HAVE_SCHED_YIELD) && defined(STARPU_HAVE_HELGRIND_H)
#define STARPU_UYIELD() do { if (RUNNING_ON_VALGRIND) sched_yield(); else _STARPU_UYIELD(); } while (0)
#else
#define STARPU_UYIELD() _STARPU_UYIELD()
#endif

#ifdef STARPU_VERBOSE
#  define _STARPU_DEBUG(fmt, ...) do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%s] " fmt ,__starpu_func__ ,## __VA_ARGS__); fflush(stderr); }} while(0)
#else
#  define _STARPU_DEBUG(fmt, ...) do { } while (0)
#endif

#ifdef STARPU_VERBOSE0
#  define _STARPU_LOG_IN()             do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%ld][%s] -->\n", pthread_self(), __starpu_func__ ); }} while(0)
#  define _STARPU_LOG_OUT()            do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%ld][%s] <--\n", pthread_self(), __starpu_func__ ); }} while(0)
#  define _STARPU_LOG_OUT_TAG(outtag)  do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%ld][%s] <-- (%s)\n", pthread_self(), __starpu_func__, outtag); }} while(0)
#else
#  define _STARPU_LOG_IN()
#  define _STARPU_LOG_OUT()
#  define _STARPU_LOG_OUT_TAG(outtag)
#endif

#define _STARPU_DISP(fmt, ...) do { if (!getenv("STARPU_SILENT")) {fprintf(stderr, "[starpu][%s] " fmt ,__starpu_func__ ,## __VA_ARGS__); }} while(0)
#define _STARPU_ERROR(fmt, ...)                                                  \
	do {                                                                          \
                fprintf(stderr, "\n\n[starpu][%s] Error: " fmt ,__starpu_func__ ,## __VA_ARGS__);    \
		fprintf(stderr, "\n\n");					      \
		STARPU_ABORT();                                                       \
	} while (0)


#define _STARPU_IS_ZERO(a) (fpclassify(a) == FP_ZERO)

int _starpu_mkpath(const char *s, mode_t mode);
void _starpu_mkpath_and_check(const char *s, mode_t mode);
char *_starpu_get_home_path(void);
void _starpu_gethostname(char *hostname, size_t size);

/* If FILE is currently on a comment line, eat it.  */
void _starpu_drop_comments(FILE *f);

struct _starpu_job;
/* Returns the symbol associated to that job if any. */
const char *_starpu_job_get_model_name(struct _starpu_job *j);

struct starpu_codelet;
/* Returns the symbol associated to that job if any. */
const char *_starpu_codelet_get_model_name(struct starpu_codelet *cl);

int _starpu_check_mutex_deadlock(starpu_pthread_mutex_t *mutex);

#endif // __COMMON_UTILS_H__
