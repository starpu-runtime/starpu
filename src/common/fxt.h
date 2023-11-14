/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Joris Pablo
 * Copyright (C) 2018,2020  Federal University of Rio Grande do Sul (UFRGS)
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

#ifndef __FXT_H__
#define __FXT_H__


/** @file */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  1 /* ou _BSD_SOURCE ou _SVID_SOURCE */
#endif

#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <common/utils.h>
#include <starpu.h>

#ifdef STARPU_USE_FXT
#include <fxt/fxt.h>
#include <fxt/fut.h>
#endif

#pragma GCC visibility push(hidden)

/* some key to identify the worker kind */
#define _STARPU_FUT_WORKER_KEY(kind) (kind + 0x100)
#define _STARPU_FUT_KEY_WORKER(key) (key - 0x100)

#define _STARPU_FUT_WORKER_INIT_START	0x5100
#define _STARPU_FUT_WORKER_INIT_END	0x5101

#define	_STARPU_FUT_START_CODELET_BODY	0x5102
#define	_STARPU_FUT_END_CODELET_BODY	0x5103

#define _STARPU_FUT_JOB_PUSH		0x5104
#define _STARPU_FUT_JOB_POP		0x5105

#define _STARPU_FUT_UPDATE_TASK_CNT	0x5106

#define _STARPU_FUT_START_FETCH_INPUT_ON_TID	0x5107
#define _STARPU_FUT_END_FETCH_INPUT_ON_TID	0x5108
#define _STARPU_FUT_START_PUSH_OUTPUT_ON_TID	0x5109
#define _STARPU_FUT_END_PUSH_OUTPUT_ON_TID	0x5110

#define _STARPU_FUT_TAG		0x5111
#define _STARPU_FUT_TAG_DEPS	0x5112

#define _STARPU_FUT_TASK_DEPS		0x5113

#define _STARPU_FUT_DATA_COPY		0x5114
#define _STARPU_FUT_WORK_STEALING	0x5115

#define _STARPU_FUT_WORKER_DEINIT_START	0x5116
#define _STARPU_FUT_WORKER_DEINIT_END	0x5117

#define _STARPU_FUT_WORKER_SLEEP_START	0x5118
#define _STARPU_FUT_WORKER_SLEEP_END	0x5119

#define _STARPU_FUT_TASK_SUBMIT		0x511a
#define _STARPU_FUT_CODELET_DATA_HANDLE	0x511b

#define _STARPU_FUT_MODEL_NAME		0x511c

#define _STARPU_FUT_DATA_NAME		0x511d
#define _STARPU_FUT_DATA_COORDINATES	0x511e
#define _STARPU_FUT_HANDLE_DATA_UNREGISTER	0x511f

#define _STARPU_FUT_CODELET_DATA_HANDLE_NUMA_ACCESS	0x5120

#define	_STARPU_FUT_NEW_MEM_NODE	0x5122

#define	_STARPU_FUT_START_CALLBACK	0x5123
#define	_STARPU_FUT_END_CALLBACK	0x5124

#define	_STARPU_FUT_TASK_DONE		0x5125
#define	_STARPU_FUT_TAG_DONE		0x5126

#define	_STARPU_FUT_START_ALLOC		0x5127
#define	_STARPU_FUT_END_ALLOC		0x5128

#define	_STARPU_FUT_START_ALLOC_REUSE	0x5129
#define	_STARPU_FUT_END_ALLOC_REUSE	0x5130

#define	_STARPU_FUT_USED_MEM	0x512a

#define _STARPU_FUT_TASK_NAME	0x512b

#define _STARPU_FUT_DATA_WONT_USE	0x512c

#define _STARPU_FUT_TASK_COLOR	0x512d

#define _STARPU_FUT_DATA_DOING_WONT_USE	0x512e

#define _STARPU_FUT_TASK_LINE	0x512f

#define	_STARPU_FUT_START_MEMRECLAIM	0x5131
#define	_STARPU_FUT_END_MEMRECLAIM	0x5132

#define	_STARPU_FUT_START_DRIVER_COPY	0x5133
#define	_STARPU_FUT_END_DRIVER_COPY	0x5134

#define	_STARPU_FUT_START_DRIVER_COPY_ASYNC	0x5135
#define	_STARPU_FUT_END_DRIVER_COPY_ASYNC	0x5136

#define	_STARPU_FUT_START_PROGRESS_ON_TID	0x5137
#define	_STARPU_FUT_END_PROGRESS_ON_TID		0x5138

#define _STARPU_FUT_USER_EVENT		0x5139

#define _STARPU_FUT_SET_PROFILING	0x513a

#define _STARPU_FUT_TASK_WAIT_FOR_ALL	0x513b

#define _STARPU_FUT_EVENT		0x513c
#define _STARPU_FUT_THREAD_EVENT	0x513d

#define	_STARPU_FUT_CODELET_DETAILS	0x513e
#define	_STARPU_FUT_CODELET_DATA	0x513f

#define _STARPU_FUT_LOCKING_MUTEX	0x5140
#define _STARPU_FUT_MUTEX_LOCKED	0x5141

#define _STARPU_FUT_UNLOCKING_MUTEX	0x5142
#define _STARPU_FUT_MUTEX_UNLOCKED	0x5143

#define _STARPU_FUT_TRYLOCK_MUTEX	0x5144

#define _STARPU_FUT_RDLOCKING_RWLOCK	0x5145
#define _STARPU_FUT_RWLOCK_RDLOCKED	0x5146

#define _STARPU_FUT_WRLOCKING_RWLOCK	0x5147
#define _STARPU_FUT_RWLOCK_WRLOCKED	0x5148

#define _STARPU_FUT_UNLOCKING_RWLOCK	0x5149
#define _STARPU_FUT_RWLOCK_UNLOCKED	0x514a

#define _STARPU_FUT_LOCKING_SPINLOCK	0x514b
#define _STARPU_FUT_SPINLOCK_LOCKED	0x514c

#define _STARPU_FUT_UNLOCKING_SPINLOCK	0x514d
#define _STARPU_FUT_SPINLOCK_UNLOCKED	0x514e

#define _STARPU_FUT_TRYLOCK_SPINLOCK	0x514f

#define _STARPU_FUT_COND_WAIT_BEGIN	0x5150
#define _STARPU_FUT_COND_WAIT_END	0x5151

#define _STARPU_FUT_MEMORY_FULL		0x5152

#define _STARPU_FUT_DATA_LOAD 		0x5153

#define _STARPU_FUT_START_UNPARTITION_ON_TID 0x5154
#define _STARPU_FUT_END_UNPARTITION_ON_TID 0x5155

#define	_STARPU_FUT_START_FREE		0x5156
#define	_STARPU_FUT_END_FREE		0x5157

#define	_STARPU_FUT_START_WRITEBACK	0x5158
#define	_STARPU_FUT_END_WRITEBACK	0x5159

#define _STARPU_FUT_SCHED_COMPONENT_PUSH_PRIO 	0x515a
#define _STARPU_FUT_SCHED_COMPONENT_POP_PRIO 	0x515b

#define	_STARPU_FUT_START_WRITEBACK_ASYNC	0x515c
#define	_STARPU_FUT_END_WRITEBACK_ASYNC		0x515d

#define	_STARPU_FUT_HYPERVISOR_BEGIN    0x5160
#define	_STARPU_FUT_HYPERVISOR_END	0x5161

#define _STARPU_FUT_BARRIER_WAIT_BEGIN		0x5162
#define _STARPU_FUT_BARRIER_WAIT_END		0x5163

#define _STARPU_FUT_WORKER_SCHEDULING_START	0x5164
#define _STARPU_FUT_WORKER_SCHEDULING_END	0x5165
#define _STARPU_FUT_WORKER_SCHEDULING_PUSH	0x5166
#define _STARPU_FUT_WORKER_SCHEDULING_POP	0x5167

#define	_STARPU_FUT_START_EXECUTING	0x5168
#define	_STARPU_FUT_END_EXECUTING	0x5169

#define _STARPU_FUT_SCHED_COMPONENT_NEW		0x516a
#define _STARPU_FUT_SCHED_COMPONENT_CONNECT	0x516b
#define _STARPU_FUT_SCHED_COMPONENT_PUSH	0x516c
#define _STARPU_FUT_SCHED_COMPONENT_PULL	0x516d

#define _STARPU_FUT_TASK_SUBMIT_START	0x516e
#define _STARPU_FUT_TASK_SUBMIT_END	0x516f

#define _STARPU_FUT_TASK_BUILD_START	0x5170
#define _STARPU_FUT_TASK_BUILD_END	0x5171

#define _STARPU_FUT_TASK_MPI_DECODE_START	0x5172
#define _STARPU_FUT_TASK_MPI_DECODE_END		0x5173

#define _STARPU_FUT_TASK_MPI_PRE_START	0x5174
#define _STARPU_FUT_TASK_MPI_PRE_END	0x5175

#define _STARPU_FUT_TASK_MPI_POST_START	0x5176
#define _STARPU_FUT_TASK_MPI_POST_END	0x5177

#define _STARPU_FUT_TASK_WAIT_START	0x5178
#define _STARPU_FUT_TASK_WAIT_END	0x5179

#define _STARPU_FUT_TASK_WAIT_FOR_ALL_START	0x517a
#define _STARPU_FUT_TASK_WAIT_FOR_ALL_END	0x517b

#define _STARPU_FUT_HANDLE_DATA_REGISTER	0x517c

#define _STARPU_FUT_START_FETCH_INPUT	0x517e
#define _STARPU_FUT_END_FETCH_INPUT	0x517f

#define _STARPU_FUT_TASK_THROTTLE_START	0x5180
#define _STARPU_FUT_TASK_THROTTLE_END	0x5181

#define _STARPU_FUT_DATA_STATE_INVALID 0x5182
#define _STARPU_FUT_DATA_STATE_OWNER      0x5183
#define _STARPU_FUT_DATA_STATE_SHARED     0x5184

#define _STARPU_FUT_DATA_REQUEST_CREATED   0x5185
#define _STARPU_FUT_PAPI_TASK_EVENT_VALUE   0x5186
#define _STARPU_FUT_TASK_EXCLUDE_FROM_DAG   0x5187

#define _STARPU_FUT_TASK_END_DEP	0x5188

#ifdef STARPU_BUBBLE
#define _STARPU_FUT_TASK_BUBBLE		0x5189
#endif

#define	_STARPU_FUT_START_PARALLEL_SYNC	0x518a
#define	_STARPU_FUT_END_PARALLEL_SYNC	0x518b

/* Predefined FUT key masks */
#define _STARPU_FUT_KEYMASK_META           FUT_KEYMASK0
#define _STARPU_FUT_KEYMASK_USER           FUT_KEYMASK1
#define _STARPU_FUT_KEYMASK_TASK           FUT_KEYMASK2
#define _STARPU_FUT_KEYMASK_TASK_VERBOSE   FUT_KEYMASK3
#define _STARPU_FUT_KEYMASK_DATA           FUT_KEYMASK4
#define _STARPU_FUT_KEYMASK_DATA_VERBOSE   FUT_KEYMASK5
#define _STARPU_FUT_KEYMASK_WORKER         FUT_KEYMASK6
#define _STARPU_FUT_KEYMASK_WORKER_VERBOSE FUT_KEYMASK7
#define _STARPU_FUT_KEYMASK_DSM            FUT_KEYMASK8
#define _STARPU_FUT_KEYMASK_DSM_VERBOSE    FUT_KEYMASK9
#define _STARPU_FUT_KEYMASK_SCHED          FUT_KEYMASK10
#define _STARPU_FUT_KEYMASK_SCHED_VERBOSE  FUT_KEYMASK11
#define _STARPU_FUT_KEYMASK_LOCK           FUT_KEYMASK12
#define _STARPU_FUT_KEYMASK_LOCK_VERBOSE   FUT_KEYMASK13
#define _STARPU_FUT_KEYMASK_EVENT          FUT_KEYMASK14
#define _STARPU_FUT_KEYMASK_EVENT_VERBOSE  FUT_KEYMASK15
#define _STARPU_FUT_KEYMASK_MPI            FUT_KEYMASK16
#define _STARPU_FUT_KEYMASK_MPI_VERBOSE    FUT_KEYMASK17
#define _STARPU_FUT_KEYMASK_HYP            FUT_KEYMASK18
#define _STARPU_FUT_KEYMASK_HYP_VERBOSE    FUT_KEYMASK19
#define _STARPU_FUT_KEYMASK_TASK_VERBOSE_EXTRA   FUT_KEYMASK20
#define _STARPU_FUT_KEYMASK_MPI_VERBOSE_EXTRA   FUT_KEYMASK21
/* When doing modifications to keymasks:
 * - also adapt _starpu_profile_get_user_keymask() in src/common/fxt.c
 * - adapt KEYMASKALL_DEFAULT in src/common/fxt.c
 * - adapt the documentation in 501_environment_variable.doxy and/or
 *   380_offline_performance_tools.doxy */

extern unsigned long _starpu_job_cnt;

static inline unsigned long _starpu_fxt_get_job_id(void)
{
	unsigned long ret = STARPU_ATOMIC_ADDL(&_starpu_job_cnt, 1);
	STARPU_ASSERT_MSG(ret != 0, "Oops, job_id wrapped! There are too many tasks for tracking them for profiling");
	return ret;
}

#ifdef STARPU_USE_FXT

/* Some versions of FxT do not include the declaration of the function */
#ifdef HAVE_ENABLE_FUT_FLUSH
#if !HAVE_DECL_ENABLE_FUT_FLUSH
void enable_fut_flush();
#endif
#endif
#ifdef HAVE_FUT_SET_FILENAME
#if !HAVE_DECL_FUT_SET_FILENAME
void fut_set_filename(char *filename);
#endif
#endif

extern int _starpu_fxt_started STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;
extern int _starpu_fxt_willstart STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;
extern starpu_pthread_mutex_t _starpu_fxt_started_mutex STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;
extern starpu_pthread_cond_t _starpu_fxt_started_cond STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

/** Wait until FXT is started (or not). Returns if FXT was started */
static inline int _starpu_fxt_wait_initialisation()
{
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_fxt_started_mutex);
	while (_starpu_fxt_willstart && !_starpu_fxt_started)
		STARPU_PTHREAD_COND_WAIT(&_starpu_fxt_started_cond, &_starpu_fxt_started_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_fxt_started_mutex);

	return _starpu_fxt_started;
}

extern unsigned long _starpu_submit_order;

static inline unsigned long _starpu_fxt_get_submit_order(void)
{
	unsigned long ret = STARPU_ATOMIC_ADDL(&_starpu_submit_order, 1);
	STARPU_ASSERT_MSG(_starpu_submit_order != 0, "Oops, submit_order wrapped! There are too many tasks for tracking them for profiling");
	return ret;
}

long _starpu_gettid(void) STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

int _starpu_generate_paje_trace_read_option(const char *option, struct starpu_fxt_options *options) STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

/** Initialize the FxT library. */
void _starpu_fxt_init_profiling(uint64_t trace_buffer_size);

/** Stop the FxT library, and generate the trace file. */
void _starpu_stop_fxt_profiling(void);

/** In case we use MPI, tell the profiling system how many processes are used. */
void _starpu_profiling_set_mpi_worldsize(int worldsize) STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

/** Generate the trace file. Used when catching signals SIGINT and SIGSEGV */
void _starpu_fxt_dump_file(void);

#ifdef FUT_NEEDS_COMMIT
#define _STARPU_FUT_COMMIT(size) fut_commitstampedbuffer(size)
#else
#define _STARPU_FUT_COMMIT(size) do { } while (0)
#endif

#ifdef FUT_RAW_ALWAYS_PROBE1STR
#define _STARPU_FUT_ALWAYS_PROBE1STR(CODE, P1, str) FUT_RAW_ALWAYS_PROBE1STR(CODE, P1, str)
#else
#define _STARPU_FUT_ALWAYS_PROBE1STR(CODE, P1, str)	\
do {									\
    if(STARPU_UNLIKELY(fut_active)) { \
	/* No more than FXT_MAX_PARAMS args are allowed */		\
	/* we add a \0 just in case ... */				\
	size_t len = STARPU_MIN(strlen(str)+1, (FXT_MAX_PARAMS - 1)*sizeof(unsigned long));\
	unsigned nbargs_str = (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	unsigned nbargs = 1 + nbargs_str;				\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *futargs =					\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(futargs++) = (unsigned long)(P1);				\
	snprintf((char *)futargs, len, "%s", str);			\
	((char *)futargs)[len - 1] = '\0';				\
	_STARPU_FUT_COMMIT(total_len);					\
    }} while (0)
#endif

#ifdef FUT_FULL_PROBE1STR
#define _STARPU_FUT_FULL_PROBE1STR(KEYMASK, CODE, P1, str) FUT_FULL_PROBE1STR(CODE, P1, str)
#else
/** Sometimes we need something a little more specific than the wrappers from
 * FxT: these macro permit to put add an event with 3 (or 4) numbers followed
 * by a string. */
#define _STARPU_FUT_FULL_PROBE1STR(KEYMASK, CODE, P1, str)		\
do {									\
    if (STARPU_UNLIKELY(KEYMASK & fut_active)) {			\
	_STARPU_FUT_ALWAYS_PROBE1STR(CODE, P1, str);		\
    }									\
} while (0)
#endif

#ifdef FUT_ALWAYS_PROBE2STR
#define _STARPU_FUT_ALWAYS_PROBE2STR(CODE, P1, P2, str) FUT_RAW_ALWAYS_PROBE2STR(CODE, P1, P2, str)
#else
#define _STARPU_FUT_ALWAYS_PROBE2STR(CODE, P1, P2, str)			\
do {									\
	/* No more than FXT_MAX_PARAMS args are allowed */		\
	/* we add a \0 just in case ... */				\
	size_t len = STARPU_MIN(strlen(str)+1, (FXT_MAX_PARAMS - 2)*sizeof(unsigned long));\
	unsigned nbargs_str = (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	unsigned nbargs = 2 + nbargs_str;				\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *futargs =					\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(futargs++) = (unsigned long)(P1);				\
	*(futargs++) = (unsigned long)(P2);				\
	snprintf((char *)futargs, len, "%s", str);			\
	((char *)futargs)[len - 1] = '\0';				\
	_STARPU_FUT_COMMIT(total_len);					\
} while (0)
#endif

#ifdef FUT_FULL_PROBE2STR
#define _STARPU_FUT_FULL_PROBE2STR(KEYMASK, CODE, P1, P2, str) FUT_FULL_PROBE2STR(CODE, P1, P2, str)
#else
#define _STARPU_FUT_FULL_PROBE2STR(KEYMASK, CODE, P1, P2, str)		\
do {									\
    if (STARPU_UNLIKELY(KEYMASK & fut_active)) {			\
	_STARPU_FUT_ALWAYS_PROBE2STR(CODE, P1, P2, str);		\
    }									\
} while (0)
#endif

#ifdef FUT_ALWAYS_PROBE3STR
#define _STARPU_FUT_ALWAYS_PROBE3STR(CODE, P1, P2, P3, str) FUT_RAW_ALWAYS_PROBE3STR(CODE, P1, P2, P3, str)
#else
#define _STARPU_FUT_ALWAYS_PROBE3STR(CODE, P1, P2, P3, str)			\
do {									\
	/* No more than FXT_MAX_PARAMS args are allowed */		\
	/* we add a \0 just in case ... */				\
	size_t len = STARPU_MIN(strlen(str)+1, (FXT_MAX_PARAMS - 3)*sizeof(unsigned long));\
	unsigned nbargs_str = (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	unsigned nbargs = 3 + nbargs_str;				\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *futargs =					\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(futargs++) = (unsigned long)(P1);				\
	*(futargs++) = (unsigned long)(P2);				\
	*(futargs++) = (unsigned long)(P3);				\
	snprintf((char *)futargs, len, "%s", str);			\
	((char *)futargs)[len - 1] = '\0';				\
	_STARPU_FUT_COMMIT(total_len);					\
} while (0)
#endif

#ifdef FUT_FULL_PROBE3STR
#define _STARPU_FUT_FULL_PROBE3STR(KEYMASK, CODE, P1, P2, P3, str) FUT_FULL_PROBE3STR(CODE, P1, P2, P3, str)
#else
#define _STARPU_FUT_FULL_PROBE3STR(KEYMASK, CODE, P1, P2, P3, str)		\
do {									\
    if (STARPU_UNLIKELY(KEYMASK & fut_active)) {			\
	_STARPU_FUT_ALWAYS_PROBE3STR(CODE, P1, P2, P3, str);	\
    }									\
} while (0)
#endif

#ifdef FUT_ALWAYS_PROBE4STR
#define _STARPU_FUT_ALWAYS_PROBE4STR(CODE, P1, P2, P3, P4, str) FUT_RAW_ALWAYS_PROBE4STR(CODE, P1, P2, P3, P4, str)
#else
#define _STARPU_FUT_ALWAYS_PROBE4STR(CODE, P1, P2, P3, P4, str)		\
do {									\
	/* No more than FXT_MAX_PARAMS args are allowed */		\
	/* we add a \0 just in case ... */				\
	size_t len = STARPU_MIN(strlen(str)+1, (FXT_MAX_PARAMS - 4)*sizeof(unsigned long));\
	unsigned nbargs_str = (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	unsigned nbargs = 4 + nbargs_str;				\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *futargs =						\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(futargs++) = (unsigned long)(P1);				\
	*(futargs++) = (unsigned long)(P2);				\
	*(futargs++) = (unsigned long)(P3);				\
	*(futargs++) = (unsigned long)(P4);				\
	snprintf((char *)futargs, len, "%s", str);			\
	((char *)futargs)[len - 1] = '\0';				\
	_STARPU_FUT_COMMIT(total_len);					\
} while (0)
#endif

#ifdef FUT_FULL_PROBE4STR
#define _STARPU_FUT_FULL_PROBE4STR(KEYMASK, CODE, P1, P2, P3, P4, str) FUT_FULL_PROBE4STR(CODE, P1, P2, P3, P4, str)
#else
#define _STARPU_FUT_FULL_PROBE4STR(KEYMASK, CODE, P1, P2, P3, P4, str)		\
do {									\
    if (STARPU_UNLIKELY(KEYMASK & fut_active)) {			\
	_STARPU_FUT_ALWAYS_PROBE4STR(CODE, P1, P2, P3, P4, str);	\
    }									\
} while (0)
#endif

#ifdef FUT_ALWAYS_PROBE5STR
#define _STARPU_FUT_ALWAYS_PROBE5STR(CODE, P1, P2, P3, P4, P5, str) FUT_RAW_ALWAYS_PROBE5STR(CODE, P1, P2, P3, P4, P5, str)
#else
#define _STARPU_FUT_ALWAYS_PROBE5STR(CODE, P1, P2, P3, P4, P5, str)		\
do {									\
	/* No more than FXT_MAX_PARAMS args are allowed */		\
	/* we add a \0 just in case ... */				\
	size_t len = STARPU_MIN(strlen(str)+1, (FXT_MAX_PARAMS - 5)*sizeof(unsigned long));\
	unsigned nbargs_str = (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	unsigned nbargs = 5 + nbargs_str;				\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *futargs =					\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(futargs++) = (unsigned long)(P1);				\
	*(futargs++) = (unsigned long)(P2);				\
	*(futargs++) = (unsigned long)(P3);				\
	*(futargs++) = (unsigned long)(P4);				\
	*(futargs++) = (unsigned long)(P5);				\
	snprintf((char *)futargs, len, "%s", str);			\
	((char *)futargs)[len - 1] = '\0';				\
	_STARPU_FUT_COMMIT(total_len);					\
} while (0)
#endif

#ifdef FUT_FULL_PROBE5STR
#define _STARPU_FUT_FULL_PROBE5STR(KEYMASK, CODE, P1, P2, P3, P4, P5, str) FUT_FULL_PROBE5STR(CODE, P1, P2, P3, P4, P5, str)
#else
#define _STARPU_FUT_FULL_PROBE5STR(KEYMASK, CODE, P1, P2, P3, P4, P5, str)		\
do {									\
    if (STARPU_UNLIKELY(KEYMASK & fut_active)) {			\
	_STARPU_FUT_ALWAYS_PROBE5STR(CODE, P1, P2, P3, P4, P5, str);	\
    }									\
} while (0)
#endif

#ifdef FUT_ALWAYS_PROBE6STR
#define _STARPU_FUT_ALWAYS_PROBE6STR(CODE, P1, P2, P3, P4, P5, P6, str) FUT_RAW_ALWAYS_PROBE6STR(CODE, P1, P2, P3, P4, P5, P6, str)
#else
#define _STARPU_FUT_ALWAYS_PROBE6STR(CODE, P1, P2, P3, P4, P5, P6, str)	\
do {									\
	/* No more than FXT_MAX_PARAMS args are allowed */		\
	/* we add a \0 just in case ... */				\
	size_t len = STARPU_MIN(strlen(str)+1, (FXT_MAX_PARAMS - 6)*sizeof(unsigned long));\
	unsigned nbargs_str = (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	unsigned nbargs = 6 + nbargs_str;				\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *futargs =					\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(futargs++) = (unsigned long)(P1);				\
	*(futargs++) = (unsigned long)(P2);				\
	*(futargs++) = (unsigned long)(P3);				\
	*(futargs++) = (unsigned long)(P4);				\
	*(futargs++) = (unsigned long)(P5);				\
	*(futargs++) = (unsigned long)(P6);				\
	snprintf((char *)futargs, len, "%s", str);			\
	((char *)futargs)[len - 1] = '\0';				\
	_STARPU_FUT_COMMIT(total_len);					\
} while (0)
#endif

#ifdef FUT_FULL_PROBE6STR
#define _STARPU_FUT_FULL_PROBE6STR(KEYMASK, CODE, P1, P2, P3, P4, P5, P6, str) FUT_FULL_PROBE6STR(CODE, P1, P2, P3, P4, P5, P6, str)
#else
#define _STARPU_FUT_FULL_PROBE6STR(KEYMASK, CODE, P1, P2, P3, P4, P5, P6, str)		\
do {									\
    if (STARPU_UNLIKELY(KEYMASK & fut_active)) {			\
	_STARPU_FUT_ALWAYS_PROBE6STR(CODE, P1, P2, P3, P4, P5, P6, str);	\
    }									\
} while (0)
#endif

#ifdef FUT_ALWAYS_PROBE7STR
#define _STARPU_FUT_ALWAYS_PROBE7STR(CODE, P1, P2, P3, P4, P5, P6, P7, str) FUT_RAW_ALWAYS_PROBE7STR(CODE, P1, P2, P3, P4, P5, P6, P7, str)
#else
#define _STARPU_FUT_ALWAYS_PROBE7STR(CODE, P1, P2, P3, P4, P5, P6, P7, str)	\
do {									\
	/* No more than FXT_MAX_PARAMS args are allowed */		\
	/* we add a \0 just in case ... */				\
	size_t len = STARPU_MIN(strlen(str)+1, (FXT_MAX_PARAMS - 7)*sizeof(unsigned long));\
	unsigned nbargs_str = (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	unsigned nbargs = 7 + nbargs_str;				\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *futargs =					\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(futargs++) = (unsigned long)(P1);				\
	*(futargs++) = (unsigned long)(P2);				\
	*(futargs++) = (unsigned long)(P3);				\
	*(futargs++) = (unsigned long)(P4);				\
	*(futargs++) = (unsigned long)(P5);				\
	*(futargs++) = (unsigned long)(P6);				\
	*(futargs++) = (unsigned long)(P7);				\
	snprintf((char *)futargs, len, "%s", str);			\
	((char *)futargs)[len - 1] = '\0';				\
	_STARPU_FUT_COMMIT(total_len);					\
} while (0)
#endif

#ifdef FUT_FULL_PROBE7STR
#define _STARPU_FUT_FULL_PROBE7STR(KEYMASK, CODE, P1, P2, P3, P4, P5, P6, P7, str) FUT_FULL_PROBE7STR(CODE, P1, P2, P3, P4, P5, P6, P7, str)
#else
#define _STARPU_FUT_FULL_PROBE7STR(KEYMASK, CODE, P1, P2, P3, P4, P5, P6, P7, str)		\
do {									\
    if (STARPU_UNLIKELY(KEYMASK & fut_active)) {			\
	_STARPU_FUT_ALWAYS_PROBE7STR(CODE, P1, P2, P3, P4, P5, P6, P7, str);	\
    }									\
} while (0)
#endif

#ifndef FUT_RAW_PROBE7
#define FUT_RAW_PROBE7(CODE,P1,P2,P3,P4,P5,P6,P7) do {		\
		if(STARPU_UNLIKELY(fut_active)) {			\
			unsigned long *__args __attribute__((unused))=	\
				fut_getstampedbuffer(CODE,		\
						     FUT_SIZE(7)); \
			*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);*(__args++)=(unsigned long)(P4);*(__args++)=(unsigned long)(P5);*(__args++)=(unsigned long)(P6);*(__args++)=(unsigned long)(P7);				\
			_STARPU_FUT_COMMIT(FUT_SIZE(7));		\
		}							\
	} while (0)
#endif

#ifndef FUT_RAW_ALWAYS_PROBE1
#define FUT_RAW_ALWAYS_PROBE1(CODE,P1) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(1)); \
		*(__args++)=(unsigned long)(P1); \
		fut_commitstampedbuffer(FUT_SIZE(1)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE1(CODE,P1) do { \
        FUT_RAW_ALWAYS_PROBE1(FUT_CODE(CODE, 1),P1); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE2
#define FUT_RAW_ALWAYS_PROBE2(CODE,P1,P2) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(2)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2); \
		fut_commitstampedbuffer(FUT_SIZE(2)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE2(CODE,P1,P2) do { \
        FUT_RAW_ALWAYS_PROBE2(FUT_CODE(CODE, 2),P1,P2); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE3
#define FUT_RAW_ALWAYS_PROBE3(CODE,P1,P2,P3) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(3)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);				\
		fut_commitstampedbuffer(FUT_SIZE(3)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE3(CODE,P1,P2,P3) do { \
        FUT_RAW_ALWAYS_PROBE3(FUT_CODE(CODE, 3),P1,P2,P3); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE4
#define FUT_RAW_ALWAYS_PROBE4(CODE,P1,P2,P3,P4) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(4)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);*(__args++)=(unsigned long)(P4);				\
		fut_commitstampedbuffer(FUT_SIZE(4)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE4(CODE,P1,P2,P3,P4) do { \
        FUT_RAW_ALWAYS_PROBE4(FUT_CODE(CODE, 4),P1,P2,P3,P4); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE5
#define FUT_RAW_ALWAYS_PROBE5(CODE,P1,P2,P3,P4,P5) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(5)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);*(__args++)=(unsigned long)(P4);*(__args++)=(unsigned long)(P5);				\
		fut_commitstampedbuffer(FUT_SIZE(5)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE5(CODE,P1,P2,P3,P4,P5) do { \
        FUT_RAW_ALWAYS_PROBE5(FUT_CODE(CODE, 5),P1,P2,P3,P4,P5); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE6
#define FUT_RAW_ALWAYS_PROBE6(CODE,P1,P2,P3,P4,P5,P6) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(6)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);*(__args++)=(unsigned long)(P4);*(__args++)=(unsigned long)(P5);*(__args++)=(unsigned long)(P6);				\
		fut_commitstampedbuffer(FUT_SIZE(6)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE6(CODE,P1,P2,P3,P4,P5,P6) do { \
        FUT_RAW_ALWAYS_PROBE6(FUT_CODE(CODE, 6),P1,P2,P3,P4,P5,P6); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE7
#define FUT_RAW_ALWAYS_PROBE7(CODE,P1,P2,P3,P4,P5,P6,P7) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(7)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);*(__args++)=(unsigned long)(P4);*(__args++)=(unsigned long)(P5);*(__args++)=(unsigned long)(P6);*(__args++)=(unsigned long)(P7);				\
		fut_commitstampedbuffer(FUT_SIZE(7)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE7(CODE,P1,P2,P3,P4,P5,P6,P7) do { \
        FUT_RAW_ALWAYS_PROBE7(FUT_CODE(CODE, 7),P1,P2,P3,P4,P5,P6,P7); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE8
#define FUT_RAW_ALWAYS_PROBE8(CODE,P1,P2,P3,P4,P5,P6,P7,P8) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(8)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);*(__args++)=(unsigned long)(P4);*(__args++)=(unsigned long)(P5);*(__args++)=(unsigned long)(P6);*(__args++)=(unsigned long)(P7);*(__args++)=(unsigned long)(P8);				\
		fut_commitstampedbuffer(FUT_SIZE(8)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE8(CODE,P1,P2,P3,P4,P5,P6,P7,P8) do { \
        FUT_RAW_ALWAYS_PROBE8(FUT_CODE(CODE, 8),P1,P2,P3,P4,P5,P6,P7,P8); \
} while (0)

#ifndef FUT_RAW_ALWAYS_PROBE9
#define FUT_RAW_ALWAYS_PROBE9(CODE,P1,P2,P3,P4,P5,P6,P7,P8,P9) do {	\
		unsigned long *__args __attribute__((unused))=	\
			fut_getstampedbuffer(CODE,		\
					     FUT_SIZE(9)); \
		*(__args++)=(unsigned long)(P1);*(__args++)=(unsigned long)(P2);*(__args++)=(unsigned long)(P3);*(__args++)=(unsigned long)(P4);*(__args++)=(unsigned long)(P5);*(__args++)=(unsigned long)(P6);*(__args++)=(unsigned long)(P7);*(__args++)=(unsigned long)(P8);*(__args++)=(unsigned long)(P9);				\
		fut_commitstampedbuffer(FUT_SIZE(9)); \
	} while (0)
#endif
#define FUT_DO_ALWAYS_PROBE9(CODE,P1,P2,P3,P4,P5,P6,P7,P8,P9) do { \
        FUT_RAW_ALWAYS_PROBE9(FUT_CODE(CODE, 9),P1,P2,P3,P4,P5,P6,P7,P8,P9); \
} while (0)

/* full probes */
#ifndef FUT_FULL_PROBE0
#define FUT_FULL_PROBE0(KEYMASK,CODE) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE0(FUT_CODE(CODE, 0)); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE1
#define FUT_FULL_PROBE1(KEYMASK,CODE,P1) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE1(FUT_CODE(CODE, 1),P1); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE2
#define FUT_FULL_PROBE2(KEYMASK,CODE,P1,P2) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE2(FUT_CODE(CODE, 2),P1,P2); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE3
#define FUT_FULL_PROBE3(KEYMASK,CODE,P1,P2,P3) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE3(FUT_CODE(CODE, 3),P1,P2,P3); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE4
#define FUT_FULL_PROBE4(KEYMASK,CODE,P1,P2,P3,P4) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE4(FUT_CODE(CODE, 4),P1,P2,P3,P4); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE5
#define FUT_FULL_PROBE5(KEYMASK,CODE,P1,P2,P3,P4,P5) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE5(FUT_CODE(CODE, 5),P1,P2,P3,P4,P5); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE6
#define FUT_FULL_PROBE6(KEYMASK,CODE,P1,P2,P3,P4,P5,P6) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE6(FUT_CODE(CODE, 6),P1,P2,P3,P4,P5,P6); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE7
#define FUT_FULL_PROBE7(KEYMASK,CODE,P1,P2,P3,P4,P5,P6,P7) do { \
        if (STARPU_UNLIKELY(KEYMASK & fut_active)) { \
                FUT_RAW_ALWAYS_PROBE7(FUT_CODE(CODE, 7),P1,P2,P3,P4,P5,P6,P7); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE8
#define FUT_FULL_PROBE8(KEYMASK,CODE,P1,P2,P3,P4,P5,P6,P7,P8) do { \
        if(KEYMASK & fut_active) { \
                FUT_RAW_ALWAYS_PROBE8(FUT_CODE(CODE, 8),P1,P2,P3,P4,P5,P6,P7,P8); \
        } \
} while(0)
#endif

#ifndef FUT_FULL_PROBE9
#define FUT_FULL_PROBE9(KEYMASK,CODE,P1,P2,P3,P4,P5,P6,P7,P8,P9) do { \
        if(KEYMASK & fut_active) { \
                FUT_RAW_ALWAYS_PROBE9(FUT_CODE(CODE, 9),P1,P2,P3,P4,P5,P6,P7,P8,P9); \
        } \
} while(0)
#endif

#define _STARPU_TRACE_NEW_MEM_NODE(nodeid)			do {\
	if (_starpu_fxt_started) \
		FUT_DO_ALWAYS_PROBE2(_STARPU_FUT_NEW_MEM_NODE, nodeid, _starpu_gettid()); \
} while (0)

#define _STARPU_TRACE_REGISTER_THREAD(cpuid)			do {\
	if (_starpu_fxt_started) \
		FUT_DO_ALWAYS_PROBE2(FUT_NEW_LWP_CODE, cpuid, _starpu_gettid()); \
} while (0)

#define _STARPU_TRACE_WORKER_INIT_START(workerkind, workerid, devid, memnode, bindid, sync)	do {\
	if (_starpu_fxt_started) \
		FUT_DO_ALWAYS_PROBE7(_STARPU_FUT_WORKER_INIT_START, _STARPU_FUT_WORKER_KEY(workerkind), workerid, devid, memnode, bindid, sync, _starpu_gettid()); \
} while (0)

#define _STARPU_TRACE_WORKER_INIT_END(__workerid)		do {\
	if (_starpu_fxt_started) \
		FUT_DO_ALWAYS_PROBE2(_STARPU_FUT_WORKER_INIT_END, _starpu_gettid(), (__workerid)); \
} while (0)

#define _STARPU_TRACE_START_CODELET_BODY(job, nimpl, perf_arch, workerid)				\
do {									\
    if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK|_STARPU_FUT_KEYMASK_TASK_VERBOSE|_STARPU_FUT_KEYMASK_DATA|_STARPU_FUT_KEYMASK_TASK_VERBOSE_EXTRA) & fut_active)) { \
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_START_CODELET_BODY, (job)->job_id, ((job)->task)->sched_ctx, workerid, starpu_worker_get_memory_node(workerid)); \
	{								\
		if ((job)->task->cl)					\
		{							\
			const int __nbuffers = STARPU_TASK_GET_NBUFFERS((job)->task);	\
			char __buf[FXT_MAX_PARAMS*sizeof(long)];	\
			int __i;					\
			for (__i = 0; __i < __nbuffers; __i++)		\
			{						\
				starpu_data_handle_t __handle = STARPU_TASK_GET_HANDLE((job)->task, __i);	\
				void *__interface = _STARPU_TASK_GET_INTERFACES((job)->task)[__i];	\
				if (__interface && __handle->ops->describe)		\
				{					\
					__handle->ops->describe(__interface, __buf, sizeof(__buf));	\
					_STARPU_FUT_FULL_PROBE1STR(_STARPU_FUT_KEYMASK_DATA, _STARPU_FUT_CODELET_DATA, workerid, __buf);	\
				}					\
				FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_CODELET_DATA_HANDLE, (job)->job_id, (__handle), _starpu_data_get_size(__handle), STARPU_TASK_GET_MODE((job)->task, __i));	\
				/* Regarding the memory location:
				 * - if the data interface doesn't provide to_pointer operation, NULL will be returned
				 *   and the location will be -1, which is fine;
				 * - we have to check whether the memory is on an actual NUMA node (and not on GPU
				 *   memory, for instance);
				 * - looking at memory location before executing the task isn't the best choice:
				 *   the page can be not allocated yet. A solution would be to get the memory
				 *   location at the end of the task, but there is no FxT probe where we iterate over
				 *   handles, after task execution.
				 * */ \
				FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_TASK_VERBOSE_EXTRA, _STARPU_FUT_CODELET_DATA_HANDLE_NUMA_ACCESS, (job)->job_id, (__i), starpu_worker_get_memory_node_kind(starpu_worker_get_type(workerid)) == STARPU_CPU_RAM && starpu_task_get_current_data_node(__i) >= 0 ? starpu_get_memory_location_bitmap(starpu_data_handle_to_pointer(__handle, (unsigned) starpu_task_get_current_data_node(__i)), starpu_data_get_size(__handle)) : -1);	\
			}						\
		}							\
		const size_t __job_size = (perf_arch == NULL) ? 0 : _starpu_job_get_data_size((job)->task->cl?(job)->task->cl->model:NULL, perf_arch, nimpl, (job)); \
		const uint32_t __job_hash = (perf_arch == NULL) ? 0 : _starpu_compute_buffers_footprint((job)->task->cl?(job)->task->cl->model:NULL, perf_arch, nimpl, (job)); \
		FUT_FULL_PROBE7(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_CODELET_DETAILS, ((job)->task)->sched_ctx, __job_size, __job_hash, (job)->task->flops / 1000 / ((job)->task->cl && job->task->cl->type != STARPU_SEQ ? j->task_size : 1), (job)->task->tag_id, workerid, ((job)->job_id)); \
	}								\
    } \
} while(0)

#define _STARPU_TRACE_END_CODELET_BODY(job, nimpl, perf_arch, workerid)			\
do {									\
    if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active)) { \
	    const size_t job_size = (perf_arch == NULL) ? 0 : _starpu_job_get_data_size((job)->task->cl?(job)->task->cl->model:NULL, perf_arch, nimpl, (job)); \
	    const uint32_t job_hash = (perf_arch == NULL) ? 0 : _starpu_compute_buffers_footprint((job)->task->cl?(job)->task->cl->model:NULL, perf_arch, nimpl, (job)); \
	    char _archname[32]="";					\
	    if (perf_arch) starpu_perfmodel_get_arch_name(perf_arch, _archname, 32, 0);	\
	    _STARPU_FUT_FULL_PROBE5STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_END_CODELET_BODY, (job)->job_id, (job_size), (job_hash), workerid, _starpu_gettid(), _archname); \
    } \
} while(0)

#define _STARPU_TRACE_START_EXECUTING(job)				\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_EXECUTING, _starpu_gettid(), (job)->job_id);

#define _STARPU_TRACE_END_EXECUTING(job)				\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_EXECUTING, _starpu_gettid(), (job)->job_id);

#define _STARPU_TRACE_START_PARALLEL_SYNC(job)				\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_PARALLEL_SYNC, _starpu_gettid(), (job)->job_id);

#define _STARPU_TRACE_END_PARALLEL_SYNC(job)				\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_PARALLEL_SYNC, _starpu_gettid(), (job)->job_id);

#define _STARPU_TRACE_START_CALLBACK(job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_CALLBACK, job, _starpu_gettid());

#define _STARPU_TRACE_END_CALLBACK(job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_CALLBACK, job, _starpu_gettid());

#define _STARPU_TRACE_JOB_PUSH(task, prio)	\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_JOB_PUSH, _starpu_get_job_associated_to_task(task)->job_id, prio, _starpu_gettid());

#define _STARPU_TRACE_JOB_POP(task, prio)	\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_JOB_POP, _starpu_get_job_associated_to_task(task)->job_id, prio, _starpu_gettid());

#define _STARPU_TRACE_UPDATE_TASK_CNT(counter)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_UPDATE_TASK_CNT, counter, _starpu_gettid())

#define _STARPU_TRACE_START_FETCH_INPUT(job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_FETCH_INPUT_ON_TID, job, _starpu_gettid());

#define _STARPU_TRACE_END_FETCH_INPUT(job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_FETCH_INPUT_ON_TID, job, _starpu_gettid());

#define _STARPU_TRACE_START_PUSH_OUTPUT(job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_PUSH_OUTPUT_ON_TID, job, _starpu_gettid());

#define _STARPU_TRACE_END_PUSH_OUTPUT(job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_PUSH_OUTPUT_ON_TID, job, _starpu_gettid());

#define _STARPU_TRACE_WORKER_END_FETCH_INPUT(job, id)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_FETCH_INPUT, job, id);

#define _STARPU_TRACE_WORKER_START_FETCH_INPUT(job, id)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_FETCH_INPUT, job, id);

#define _STARPU_TRACE_TAG(tag, job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG, tag, (job)->job_id)

#define _STARPU_TRACE_TAG_DEPS(tag_child, tag_father)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG_DEPS, tag_child, tag_father)

#define _STARPU_TRACE_TASK_DEPS(job_prev, job_succ)	\
	_STARPU_FUT_FULL_PROBE4STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_DEPS, (job_prev)->job_id, (job_succ)->job_id, (job_succ)->task->type, 1, "task")

#define _STARPU_TRACE_TASK_END_DEP(job_prev, job_succ) \
	FUT_DO_PROBE2(_STARPU_FUT_TASK_END_DEP, (job_prev)->job_id, (job_succ)->job_id)

#define _STARPU_TRACE_GHOST_TASK_DEPS(ghost_prev_id, job_succ)		\
	_STARPU_FUT_FULL_PROBE4STR(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_DEPS, (ghost_prev_id), (job_succ)->job_id, (job_succ)->task->type, 1, "ghost")

#ifdef STARPU_BUBBLE
#define _STARPU_TRACE_BUBBLE_TASK_DEPS(prev_id, job_succ)		\
	_STARPU_FUT_FULL_PROBE4STR(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_DEPS, (prev_id), (job_succ)->job_id, (job_succ)->task->type, 1, "bubble")
#endif

#define _STARPU_TRACE_TASK_EXCLUDE_FROM_DAG(job)			\
	do {								\
	unsigned exclude_from_dag = (job)->exclude_from_dag;		\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_EXCLUDE_FROM_DAG, (job)->job_id, (long unsigned)exclude_from_dag); \
} while(0)

#define _STARPU_TRACE_TASK_NAME_LINE_COLOR(job)				\
	do {								\
		_STARPU_TRACE_TASK_NAME(job);				\
		_STARPU_TRACE_TASK_LINE(job);				\
		_STARPU_TRACE_TASK_COLOR(job);				\
	} while(0)

#define _STARPU_TRACE_TASK_LINE(job)					\
	do {								\
		if ((job)->task->file)					\
			_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_LINE, (job)->job_id, (job)->task->line, (job)->task->file); \
} while(0)

#ifdef STARPU_BUBBLE
#define _STARPU_TRACE_BUBBLE(job)					\
do {								\
    if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active)) { \
	unsigned int is_bubble=(job)->is_bubble;			\
	unsigned long bubble_parent=(job)->task->bubble_parent;		\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_BUBBLE, (job)->job_id, is_bubble, bubble_parent); \
    } \
} while(0)
#endif

#define _STARPU_TRACE_TASK_NAME(job)				\
do {								\
    if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active)) { \
        const char *model_name = _starpu_job_get_model_name((job));		\
	const char *name = _starpu_job_get_task_name((job));			\
	if (name)					                        \
	{									\
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_NAME, (job)->job_id, _starpu_gettid(), name); \
	}									\
	else {									\
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_NAME, (job)->job_id, _starpu_gettid(), "unknown");\
	}									\
	if (model_name)					\
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_MODEL_NAME, (job)->job_id, _starpu_gettid(), model_name); \
    } \
} while(0)

#define _STARPU_TRACE_TASK_COLOR(job)						\
do { \
    if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active)) { \
	if ((job)->task->color != 0) \
		FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_COLOR, (job)->job_id, (job)->task->color); \
	else if ((job)->task->cl && (job)->task->cl->color != 0) \
		FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_COLOR, (job)->job_id, (job)->task->cl->color); \
    } \
} while(0)

#define _STARPU_TRACE_TASK_DONE(job)						\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_DONE, (job)->job_id, _starpu_gettid())

#define _STARPU_TRACE_TAG_DONE(tag)						\
do {										\
    if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active)) { \
        struct _starpu_job *job = (tag)->job;                                  \
        const char *model_name = _starpu_job_get_task_name((job));                       \
	if (model_name)                                                         \
	{									\
          _STARPU_FUT_FULL_PROBE3STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG_DONE, (tag)->id, _starpu_gettid(), 1, model_name); \
	}									\
	else {									\
		FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG_DONE, (tag)->id, _starpu_gettid(), 0);\
	}									\
    } \
} while(0)

#define _STARPU_TRACE_DATA_NAME(handle, name) \
	_STARPU_FUT_FULL_PROBE1STR(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_NAME, handle, name)

#define _STARPU_TRACE_DATA_COORDINATES(handle, dim, v) do {\
	switch (dim) { \
	case 1: FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0]); break; \
	case 2: FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1]); break; \
	case 3: FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1], v[2]); break; \
	case 4: FUT_FULL_PROBE6(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1], v[2], v[3]); break; \
	default: FUT_FULL_PROBE7(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1], v[2], v[3], v[4]); break; \
	} \
} while (0)

#define _STARPU_TRACE_DATA_COPY(src_node, dst_node, size)	\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_DATA_COPY, src_node, dst_node, size)

#define _STARPU_TRACE_DATA_WONT_USE(handle)						\
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DATA, _STARPU_FUT_DATA_WONT_USE, handle, _starpu_fxt_get_submit_order(), _starpu_fxt_get_job_id(), _starpu_gettid())

#define _STARPU_TRACE_DATA_DOING_WONT_USE(handle)						\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_DATA_DOING_WONT_USE, handle)

#define _STARPU_TRACE_START_DRIVER_COPY(src_node, dst_node, size, com_id, prefetch, handle) \
	FUT_FULL_PROBE6(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_DRIVER_COPY, src_node, dst_node, size, com_id, prefetch, handle)

#define _STARPU_TRACE_END_DRIVER_COPY(src_node, dst_node, size, com_id, prefetch)	\
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_DRIVER_COPY, src_node, dst_node, size, com_id, prefetch)

#define _STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_DRIVER_COPY_ASYNC, src_node, dst_node)

#define _STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_DRIVER_COPY_ASYNC, src_node, dst_node)

#define _STARPU_TRACE_WORK_STEALING(empty_q, victim_q)		\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_SCHED_VERBOSE, _STARPU_FUT_WORK_STEALING, empty_q, victim_q)

#define _STARPU_TRACE_WORKER_DEINIT_START			do {\
	if (_starpu_fxt_started) \
		FUT_DO_ALWAYS_PROBE1(_STARPU_FUT_WORKER_DEINIT_START, _starpu_gettid()); \
} while(0)

#define _STARPU_TRACE_WORKER_DEINIT_END(workerkind)		do {\
	if (_starpu_fxt_started) \
		FUT_DO_ALWAYS_PROBE2(_STARPU_FUT_WORKER_DEINIT_END, _STARPU_FUT_WORKER_KEY(workerkind), _starpu_gettid()); \
} while(0)

#define _STARPU_TRACE_WORKER_SCHEDULING_START	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_START, _starpu_gettid());

#define _STARPU_TRACE_WORKER_SCHEDULING_END	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_END, _starpu_gettid());

#define _STARPU_TRACE_WORKER_SCHEDULING_PUSH	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_PUSH, _starpu_gettid());

#define _STARPU_TRACE_WORKER_SCHEDULING_POP	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_POP, _starpu_gettid());

#define _STARPU_TRACE_WORKER_SLEEP_START	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER, _STARPU_FUT_WORKER_SLEEP_START, _starpu_gettid());

#define _STARPU_TRACE_WORKER_SLEEP_END	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER, _STARPU_FUT_WORKER_SLEEP_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_SUBMIT(job, iter, subiter)	\
	FUT_FULL_PROBE7(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_SUBMIT, (job)->job_id, iter, subiter, (job)->task->no_submitorder?0:_starpu_fxt_get_submit_order(), (job)->task->priority, (job)->task->type, _starpu_gettid());

#define _STARPU_TRACE_TASK_SUBMIT_START()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_SUBMIT_START, _starpu_gettid());

#define _STARPU_TRACE_TASK_SUBMIT_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_SUBMIT_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_THROTTLE_START()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_THROTTLE_START, _starpu_gettid());

#define _STARPU_TRACE_TASK_THROTTLE_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_THROTTLE_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_BUILD_START()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_BUILD_START, _starpu_gettid());

#define _STARPU_TRACE_TASK_BUILD_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_BUILD_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_MPI_DECODE_START()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_DECODE_START, _starpu_gettid());

#define _STARPU_TRACE_TASK_MPI_DECODE_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_DECODE_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_MPI_PRE_START()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_PRE_START, _starpu_gettid());

#define _STARPU_TRACE_TASK_MPI_PRE_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_PRE_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_MPI_POST_START()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_POST_START, _starpu_gettid());

#define _STARPU_TRACE_TASK_MPI_POST_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_POST_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_WAIT_START(job)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_START, (job)->job_id, _starpu_gettid());

#define _STARPU_TRACE_TASK_WAIT_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_END, _starpu_gettid());

#define _STARPU_TRACE_TASK_WAIT_FOR_ALL_START()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_FOR_ALL_START, _starpu_gettid());

#define _STARPU_TRACE_TASK_WAIT_FOR_ALL_END()	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_FOR_ALL_END, _starpu_gettid());

#define _STARPU_TRACE_START_ALLOC(memnode, size, handle, is_prefetch)               \
       FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_ALLOC, memnode, _starpu_gettid(), size, handle, is_prefetch);

#define _STARPU_TRACE_END_ALLOC(memnode, handle, r)            \
       FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_ALLOC, memnode, _starpu_gettid(), handle, r);

#define _STARPU_TRACE_START_ALLOC_REUSE(memnode, size, handle, is_prefetch)         \
       FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_ALLOC_REUSE, memnode, _starpu_gettid(), size, handle, is_prefetch);

#define _STARPU_TRACE_END_ALLOC_REUSE(memnode, handle, r)              \
       FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_ALLOC_REUSE, memnode, _starpu_gettid(), handle, r);

#define _STARPU_TRACE_START_FREE(memnode, size, handle)                \
       FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_FREE, memnode, _starpu_gettid(), size, handle);

#define _STARPU_TRACE_END_FREE(memnode, handle)                \
       FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_FREE, memnode, _starpu_gettid(), handle);

#define _STARPU_TRACE_START_WRITEBACK(memnode, handle)         \
       FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_WRITEBACK, memnode, _starpu_gettid(), handle);

#define _STARPU_TRACE_END_WRITEBACK(memnode, handle)           \
       FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_WRITEBACK, memnode, _starpu_gettid(), handle);

#define _STARPU_TRACE_USED_MEM(memnode,used)		\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_USED_MEM, memnode, used, _starpu_gettid());

#define _STARPU_TRACE_START_MEMRECLAIM(memnode,is_prefetch)		\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_MEMRECLAIM, memnode, is_prefetch, _starpu_gettid());

#define _STARPU_TRACE_END_MEMRECLAIM(memnode, is_prefetch)		\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_MEMRECLAIM, memnode, is_prefetch, _starpu_gettid());

#define _STARPU_TRACE_START_WRITEBACK_ASYNC(memnode)		\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_WRITEBACK_ASYNC, memnode, _starpu_gettid());

#define _STARPU_TRACE_END_WRITEBACK_ASYNC(memnode)		\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_WRITEBACK_ASYNC, memnode, _starpu_gettid());

#define _STARPU_TRACE_PAPI_TASK_EVENT(event_id, task, value)	\
	FUT_DO_PROBE3(_STARPU_FUT_PAPI_TASK_EVENT_VALUE, event_id, _starpu_get_job_associated_to_task(task)->job_id, value)

/* We skip these events because they are called so often that they cause FxT to
 * fail and make the overall trace unreadable anyway. */
#define _STARPU_TRACE_START_PROGRESS(memnode)		\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_PROGRESS_ON_TID, memnode, _starpu_gettid());

#define _STARPU_TRACE_END_PROGRESS(memnode)		\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_PROGRESS_ON_TID, memnode, _starpu_gettid());

#define _STARPU_TRACE_USER_EVENT(code)			\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_USER, _STARPU_FUT_USER_EVENT, code, _starpu_gettid());

#define _STARPU_TRACE_META(S)			\
	FUT_FULL_PROBESTR(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_EVENT,S)

#define _STARPU_TRACE_SET_PROFILING(status)		\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_SET_PROFILING, status, _starpu_gettid());

#define _STARPU_TRACE_TASK_WAIT_FOR_ALL			\
	FUT_FULL_PROBE0(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_WAIT_FOR_ALL)

#define _STARPU_TRACE_EVENT_ALWAYS(S)			do {\
	if (_starpu_fxt_started) \
		FUT_DO_ALWAYS_PROBESTR(_STARPU_FUT_EVENT,S) \
} while(0)

#define _STARPU_TRACE_EVENT(S)			\
	FUT_FULL_PROBESTR(_STARPU_FUT_KEYMASK_EVENT, _STARPU_FUT_EVENT,S)

#define _STARPU_TRACE_EVENT_VERBOSE(S)			\
	FUT_FULL_PROBESTR(_STARPU_FUT_KEYMASK_EVENT_VERBOSE, _STARPU_FUT_EVENT,S)


#define _STARPU_TRACE_THREAD_EVENT(S)			\
	_STARPU_FUT_FULL_PROBE1STR(_STARPU_FUT_KEYMASK_WORKER, _STARPU_FUT_THREAD_EVENT, _starpu_gettid(), S)

#define _STARPU_TRACE_HYPERVISOR_BEGIN()  \
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_HYP, _STARPU_FUT_HYPERVISOR_BEGIN, _starpu_gettid());

#define _STARPU_TRACE_HYPERVISOR_END() \
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_HYP, _STARPU_FUT_HYPERVISOR_END, _starpu_gettid());

#ifdef STARPU_FXT_LOCK_TRACES

#define _STARPU_TRACE_LOCKING_MUTEX()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_LOCKING_MUTEX,__LINE__,_starpu_gettid(),file); \
} while (0)

#define _STARPU_TRACE_MUTEX_LOCKED()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_MUTEX_LOCKED,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_UNLOCKING_MUTEX()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_UNLOCKING_MUTEX,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_MUTEX_UNLOCKED()	do {\
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_MUTEX_UNLOCKED,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_TRYLOCK_MUTEX()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_TRYLOCK_MUTEX,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_RDLOCKING_RWLOCK()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_RDLOCKING_RWLOCK,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_RWLOCK_RDLOCKED()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_RWLOCK_RDLOCKED,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_WRLOCKING_RWLOCK()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_WRLOCKING_RWLOCK,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_RWLOCK_WRLOCKED()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_RWLOCK_WRLOCKED,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_UNLOCKING_RWLOCK()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_UNLOCKING_RWLOCK,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_RWLOCK_UNLOCKED()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_RWLOCK_UNLOCKED,__LINE__,_starpu_gettid(),file); \
} while(0)

#define STARPU_TRACE_SPINLOCK_CONDITITION (starpu_worker_get_type(starpu_worker_get_id()) == STARPU_CUDA_WORKER)

#define _STARPU_TRACE_LOCKING_SPINLOCK(file, line)	do {\
	if (STARPU_TRACE_SPINLOCK_CONDITITION) { \
		const char *xfile; \
		xfile = strrchr(file,'/') + 1; \
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_LOCKING_SPINLOCK,line,_starpu_gettid(),xfile); \
	} \
} while(0)

#define _STARPU_TRACE_SPINLOCK_LOCKED(file, line)		do { \
	if (STARPU_TRACE_SPINLOCK_CONDITITION) { \
		const char *xfile; \
		xfile = strrchr(file,'/') + 1; \
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_SPINLOCK_LOCKED,line,_starpu_gettid(),xfile); \
	} \
} while(0)

#define _STARPU_TRACE_UNLOCKING_SPINLOCK(file, line)	do { \
	if (STARPU_TRACE_SPINLOCK_CONDITITION) { \
		const char *xfile; \
		xfile = strrchr(file,'/') + 1; \
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_UNLOCKING_SPINLOCK,line,_starpu_gettid(),xfile); \
	} \
} while(0)

#define _STARPU_TRACE_SPINLOCK_UNLOCKED(file, line)	do { \
	if (STARPU_TRACE_SPINLOCK_CONDITITION) { \
		const char *xfile; \
		xfile = strrchr(file,'/') + 1; \
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_SPINLOCK_UNLOCKED,line,_starpu_gettid(),xfile); \
	} \
} while(0)

#define _STARPU_TRACE_TRYLOCK_SPINLOCK(file, line)	do { \
	if (STARPU_TRACE_SPINLOCK_CONDITITION) { \
		const char *xfile; \
		xfile = strrchr(file,'/') + 1; \
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_TRYLOCK_SPINLOCK,line,_starpu_gettid(),xfile); \
	} \
} while(0)

#define _STARPU_TRACE_COND_WAIT_BEGIN()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_COND_WAIT_BEGIN,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_COND_WAIT_END()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_COND_WAIT_END,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_BARRIER_WAIT_BEGIN()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_BARRIER_WAIT_BEGIN,__LINE__,_starpu_gettid(),file); \
} while(0)

#define _STARPU_TRACE_BARRIER_WAIT_END()	do { \
	const char *file; \
	file = strrchr(__FILE__,'/') + 1; \
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_BARRIER_WAIT_END,__LINE__,_starpu_gettid(),file); \
} while(0)

#else // !STARPU_FXT_LOCK_TRACES

#define _STARPU_TRACE_LOCKING_MUTEX()			do {} while(0)
#define _STARPU_TRACE_MUTEX_LOCKED()			do {} while(0)
#define _STARPU_TRACE_UNLOCKING_MUTEX()			do {} while(0)
#define _STARPU_TRACE_MUTEX_UNLOCKED()			do {} while(0)
#define _STARPU_TRACE_TRYLOCK_MUTEX()			do {} while(0)
#define _STARPU_TRACE_RDLOCKING_RWLOCK()		do {} while(0)
#define _STARPU_TRACE_RWLOCK_RDLOCKED()			do {} while(0)
#define _STARPU_TRACE_WRLOCKING_RWLOCK()		do {} while(0)
#define _STARPU_TRACE_RWLOCK_WRLOCKED()			do {} while(0)
#define _STARPU_TRACE_UNLOCKING_RWLOCK()		do {} while(0)
#define _STARPU_TRACE_RWLOCK_UNLOCKED()			do {} while(0)
#define _STARPU_TRACE_LOCKING_SPINLOCK(file, line)	do {(void) file; (void)line;} while(0)
#define _STARPU_TRACE_SPINLOCK_LOCKED(file, line)	do {(void) file; (void)line;} while(0)
#define _STARPU_TRACE_UNLOCKING_SPINLOCK(file, line)	do {(void) file; (void)line;} while(0)
#define _STARPU_TRACE_SPINLOCK_UNLOCKED(file, line)	do {(void) file; (void)line;} while(0)
#define _STARPU_TRACE_TRYLOCK_SPINLOCK(file, line)	do {(void) file; (void)line;} while(0)
#define _STARPU_TRACE_COND_WAIT_BEGIN()			do {} while(0)
#define _STARPU_TRACE_COND_WAIT_END()			do {} while(0)
#define _STARPU_TRACE_BARRIER_WAIT_BEGIN()		do {} while(0)
#define _STARPU_TRACE_BARRIER_WAIT_END()		do {} while(0)

#endif // STARPU_FXT_LOCK_TRACES

#define _STARPU_TRACE_MEMORY_FULL(size)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_MEMORY_FULL,size,_starpu_gettid());

#define _STARPU_TRACE_DATA_LOAD(workerid,size)	\
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_DATA_LOAD, workerid, size);

#define _STARPU_TRACE_START_UNPARTITION(handle, memnode)		\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_UNPARTITION_ON_TID, memnode, _starpu_gettid(), handle);

#define _STARPU_TRACE_END_UNPARTITION(handle, memnode)		\
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_UNPARTITION_ON_TID, memnode, _starpu_gettid(), handle);

#define _STARPU_TRACE_SCHED_COMPONENT_PUSH_PRIO(workerid, ntasks, exp_len)		\
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_PUSH_PRIO, _starpu_gettid(), workerid, ntasks, exp_len);

#define _STARPU_TRACE_SCHED_COMPONENT_POP_PRIO(workerid, ntasks, exp_len)		\
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_POP_PRIO, _starpu_gettid(), workerid, ntasks, exp_len);

#define _STARPU_TRACE_SCHED_COMPONENT_NEW(component)		\
	if (STARPU_UNLIKELY(fut_active)) _STARPU_FUT_ALWAYS_PROBE1STR(_STARPU_FUT_SCHED_COMPONENT_NEW, component, (component)->name);

#define _STARPU_TRACE_SCHED_COMPONENT_CONNECT(parent, child)		\
	if (STARPU_UNLIKELY(fut_active)) FUT_RAW_ALWAYS_PROBE2(FUT_CODE(_STARPU_FUT_SCHED_COMPONENT_CONNECT,2), parent, child);

#define _STARPU_TRACE_SCHED_COMPONENT_PUSH(from, to, task, prio)		\
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_PUSH, _starpu_gettid(), from, to, task, prio);

#define _STARPU_TRACE_SCHED_COMPONENT_PULL(from, to, task)		\
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_PULL, _starpu_gettid(), from, to, task, (task)->priority);

#define _STARPU_TRACE_HANDLE_DATA_REGISTER(handle)	do {	\
    if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_META) & fut_active)) { \
	const size_t __data_size = handle->ops->get_size(handle); \
	const starpu_ssize_t __max_data_size = _starpu_data_get_max_size(handle); \
	char __buf[(FXT_MAX_PARAMS-4)*sizeof(long)]; \
	void *__interface = handle->per_node[0].data_interface; \
	if (handle->ops->describe) \
		handle->ops->describe(__interface, __buf, sizeof(__buf)); \
	else \
		__buf[0] = 0; \
	_STARPU_FUT_FULL_PROBE4STR(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_HANDLE_DATA_REGISTER, handle, __data_size, __max_data_size, handle->home_node, __buf); \
    } \
} while (0)

#define _STARPU_TRACE_HANDLE_DATA_UNREGISTER(handle)	\
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_DATA, _STARPU_FUT_HANDLE_DATA_UNREGISTER, handle)

//Coherency Data Traces
#define _STARPU_TRACE_DATA_STATE_INVALID(handle, node)      \
       FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_STATE_INVALID, handle, node)

#define _STARPU_TRACE_DATA_STATE_OWNER(handle, node)           \
       FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_STATE_OWNER, handle, node)

#define _STARPU_TRACE_DATA_STATE_SHARED(handle, node)          \
       FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_STATE_SHARED, handle, node)

#define _STARPU_TRACE_DATA_REQUEST_CREATED(handle, orig, dest, prio, is_pre, req)          \
       FUT_FULL_PROBE6(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_REQUEST_CREATED, orig, dest, prio, handle, is_pre, req)


#else // !STARPU_USE_FXT

/* Dummy macros in case FxT is disabled */
#define _STARPU_TRACE_NEW_MEM_NODE(nodeid)		do {(void)(nodeid);} while(0)
#define _STARPU_TRACE_REGISTER_THREAD(cpuid)		do {(void)(cpuid);} while(0)
#define _STARPU_TRACE_WORKER_INIT_START(a,b,c,d,e,f)	do {(void)(a); (void)(b); (void)(c); (void)(d); (void)(e); (void)(f);} while(0)
#define _STARPU_TRACE_WORKER_INIT_END(workerid)		do {(void)(workerid);} while(0)
#define _STARPU_TRACE_START_CODELET_BODY(job, nimpl, perf_arch, workerid) 	do {(void)(job); (void)(nimpl); (void)(perf_arch); (void)(workerid);} while(0)
#define _STARPU_TRACE_END_CODELET_BODY(job, nimpl, perf_arch, workerid)		do {(void)(job); (void)(nimpl); (void)(perf_arch); (void)(workerid);} while(0)
#define _STARPU_TRACE_START_EXECUTING(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_END_EXECUTING(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_START_PARALLEL_SYNC(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_END_PARALLEL_SYNC(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_START_CALLBACK(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_END_CALLBACK(job)		do {(void)(job);} while(0)
#define _STARPU_TRACE_JOB_PUSH(task, prio)	do {(void)(task); (void)(prio);} while(0)
#define _STARPU_TRACE_JOB_POP(task, prio)	do {(void)(task); (void)(prio);} while(0)
#define _STARPU_TRACE_UPDATE_TASK_CNT(counter)	do {(void)(counter);} while(0)
#define _STARPU_TRACE_START_FETCH_INPUT(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_END_FETCH_INPUT(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_START_PUSH_OUTPUT(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_END_PUSH_OUTPUT(job)	do {(void)(job);} while(0)
#define _STARPU_TRACE_TAG(tag, job)		do {(void)(tag); (void)(job);} while(0)
#define _STARPU_TRACE_TAG_DEPS(a, b)		do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_TASK_DEPS(a, b)		do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_TASK_END_DEP(a, b)	do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_GHOST_TASK_DEPS(a, b)	do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_TASK_EXCLUDE_FROM_DAG(a)	do {(void)(a);} while(0)
#define _STARPU_TRACE_TASK_NAME_LINE_COLOR(a)   do {(void)(a);} while(0)
#define _STARPU_TRACE_TASK_NAME(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_TASK_LINE(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_TASK_COLOR(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_TASK_DONE(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_TAG_DONE(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_DATA_NAME(a, b)		do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_DATA_COORDINATES(a, b, c)	do {(void)(a); (void)(b); (void)(c);} while(0)
#define _STARPU_TRACE_DATA_COPY(a, b, c)		do {(void)(a); (void)(b); (void)(c);} while(0)
#define _STARPU_TRACE_DATA_WONT_USE(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_DATA_DOING_WONT_USE(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_START_DRIVER_COPY(a,b,c,d,e,f)	do {(void)(a); (void)(b); (void)(c); (void)(d); (void)(e); (void)(f);} while(0)
#define _STARPU_TRACE_END_DRIVER_COPY(a,b,c,d,e)	do {(void)(a); (void)(b); (void)(c); (void)(d); (void)(e);} while(0)
#define _STARPU_TRACE_START_DRIVER_COPY_ASYNC(a,b)	do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_END_DRIVER_COPY_ASYNC(a,b)	do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_WORK_STEALING(a, b)		do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_WORKER_DEINIT_START		do {} while(0)
#define _STARPU_TRACE_WORKER_DEINIT_END(a)		do {(void)(a);} while(0)
#define _STARPU_TRACE_WORKER_SCHEDULING_START		do {} while(0)
#define _STARPU_TRACE_WORKER_SCHEDULING_END		do {} while(0)
#define _STARPU_TRACE_WORKER_SCHEDULING_PUSH		do {} while(0)
#define _STARPU_TRACE_WORKER_SCHEDULING_POP		do {} while(0)
#define _STARPU_TRACE_WORKER_SLEEP_START		do {} while(0)
#define _STARPU_TRACE_WORKER_SLEEP_END			do {} while(0)
#define _STARPU_TRACE_TASK_SUBMIT(job, a, b)			do {(void)(job); (void)(a);(void)(b);} while(0)
#define _STARPU_TRACE_TASK_SUBMIT_START()		do {} while(0)
#define _STARPU_TRACE_TASK_SUBMIT_END()			do {} while(0)
#define _STARPU_TRACE_TASK_THROTTLE_START()		do {} while(0)
#define _STARPU_TRACE_TASK_THROTTLE_END()		do {} while(0)
#define _STARPU_TRACE_TASK_BUILD_START()		do {} while(0)
#define _STARPU_TRACE_TASK_BUILD_END()			do {} while(0)
#define _STARPU_TRACE_TASK_MPI_DECODE_START()		do {} while(0)
#define _STARPU_TRACE_TASK_MPI_DECODE_END()		do {} while(0)
#define _STARPU_TRACE_TASK_MPI_PRE_START()		do {} while(0)
#define _STARPU_TRACE_TASK_MPI_PRE_END()		do {} while(0)
#define _STARPU_TRACE_TASK_MPI_POST_START()		do {} while(0)
#define _STARPU_TRACE_TASK_MPI_POST_END()		do {} while(0)
#define _STARPU_TRACE_TASK_WAIT_START(job)		do {(void)(job);} while(0)
#define _STARPU_TRACE_TASK_WAIT_END()			do {} while(0)
#define _STARPU_TRACE_TASK_WAIT_FOR_ALL_START()		do {} while(0)
#define _STARPU_TRACE_TASK_WAIT_FOR_ALL_END()		do {} while(0)
#define _STARPU_TRACE_START_ALLOC(memnode, size, handle, is_prefetch)       do {(void)(memnode); (void)(size); (void)(handle);} while(0)
#define _STARPU_TRACE_END_ALLOC(memnode, handle, r)            do {(void)(memnode); (void)(handle); (void)(r);} while(0)
#define _STARPU_TRACE_START_ALLOC_REUSE(a, size, handle, is_prefetch)       do {(void)(a); (void)(size); (void)(handle);} while(0)
#define _STARPU_TRACE_END_ALLOC_REUSE(a, handle, r)            do {(void)(a); (void)(handle); (void)(r);} while(0)
#define _STARPU_TRACE_START_FREE(memnode, size, handle)                do {(void)(memnode); (void)(size); (void)(handle);} while(0)
#define _STARPU_TRACE_END_FREE(memnode, handle)                        do {(void)(memnode); (void)(handle);} while(0)
#define _STARPU_TRACE_START_WRITEBACK(memnode, handle)         do {(void)(memnode); (void)(handle);} while(0)
#define _STARPU_TRACE_END_WRITEBACK(memnode, handle)           do {(void)(memnode); (void)(handle);} while(0)
#define _STARPU_TRACE_USED_MEM(memnode,used)		do {(void)(memnode); (void)(used);} while (0)
#define _STARPU_TRACE_START_MEMRECLAIM(memnode,is_prefetch)	do {(void)(memnode); (void)(is_prefetch);} while(0)
#define _STARPU_TRACE_END_MEMRECLAIM(memnode,is_prefetch)	do {(void)(memnode); (void)(is_prefetch);} while(0)
#define _STARPU_TRACE_START_WRITEBACK_ASYNC(memnode)	do {(void)(memnode);} while(0)
#define _STARPU_TRACE_END_WRITEBACK_ASYNC(memnode)	do {(void)(memnode);} while(0)
#define _STARPU_TRACE_START_PROGRESS(memnode)		do {(void)(memnode);} while(0)
#define _STARPU_TRACE_END_PROGRESS(memnode)		do {(void)(memnode);} while(0)
#define _STARPU_TRACE_USER_EVENT(code)			do {(void)(code);} while(0)
#define _STARPU_TRACE_SET_PROFILING(status)		do {(void)(status);} while(0)
#define _STARPU_TRACE_TASK_WAIT_FOR_ALL()		do {} while(0)
#define _STARPU_TRACE_EVENT_ALWAYS(S)				do {(void)(S);} while(0)
#define _STARPU_TRACE_EVENT(S)				do {(void)(S);} while(0)
#define _STARPU_TRACE_EVENT_VERBOSE(S)				do {(void)(S);} while(0)
#define _STARPU_TRACE_THREAD_EVENT(S)			do {(void)(S);} while(0)
#define _STARPU_TRACE_LOCKING_MUTEX()			do {} while(0)
#define _STARPU_TRACE_MUTEX_LOCKED()			do {} while(0)
#define _STARPU_TRACE_UNLOCKING_MUTEX()			do {} while(0)
#define _STARPU_TRACE_MUTEX_UNLOCKED()			do {} while(0)
#define _STARPU_TRACE_TRYLOCK_MUTEX()			do {} while(0)
#define _STARPU_TRACE_RDLOCKING_RWLOCK()		do {} while(0)
#define _STARPU_TRACE_RWLOCK_RDLOCKED()			do {} while(0)
#define _STARPU_TRACE_WRLOCKING_RWLOCK()		do {} while(0)
#define _STARPU_TRACE_RWLOCK_WRLOCKED()			do {} while(0)
#define _STARPU_TRACE_UNLOCKING_RWLOCK()		do {} while(0)
#define _STARPU_TRACE_RWLOCK_UNLOCKED()			do {} while(0)
#define _STARPU_TRACE_LOCKING_SPINLOCK(file, line)	do {(void)(file); (void)(line);} while(0)
#define _STARPU_TRACE_SPINLOCK_LOCKED(file, line)	do {(void)(file); (void)(line);} while(0)
#define _STARPU_TRACE_UNLOCKING_SPINLOCK(file, line)	do {(void)(file); (void)(line);} while(0)
#define _STARPU_TRACE_SPINLOCK_UNLOCKED(file, line)	do {(void)(file); (void)(line);} while(0)
#define _STARPU_TRACE_TRYLOCK_SPINLOCK(file, line)	do {(void)(file); (void)(line);} while(0)
#define _STARPU_TRACE_COND_WAIT_BEGIN()			do {} while(0)
#define _STARPU_TRACE_COND_WAIT_END()			do {} while(0)
#define _STARPU_TRACE_BARRIER_WAIT_BEGIN()		do {} while(0)
#define _STARPU_TRACE_BARRIER_WAIT_END()		do {} while(0)
#define _STARPU_TRACE_MEMORY_FULL(size)			do {(void)(size);} while(0)
#define _STARPU_TRACE_DATA_LOAD(workerid,size)		do {(void)(workerid); (void)(size);} while(0)
#define _STARPU_TRACE_START_UNPARTITION(handle, memnode) 	do {(void)(handle); (void)(memnode);} while(0)
#define _STARPU_TRACE_END_UNPARTITION(handle, memnode)		do {(void)(handle); (void)(memnode);} while(0)
#define _STARPU_TRACE_SCHED_COMPONENT_PUSH_PRIO(workerid, ntasks, exp_len)	do {(void)(workerid); (void)(ntasks); (void)(exp_len);} while(0)
#define _STARPU_TRACE_SCHED_COMPONENT_POP_PRIO(workerid, ntasks, exp_len)	do {(void)(workerid); (void)(ntasks); (void)(exp_len);} while(0)
#define _STARPU_TRACE_HYPERVISOR_BEGIN()        	do {} while(0)
#define _STARPU_TRACE_HYPERVISOR_END()                  do {} while(0)
#define _STARPU_TRACE_SCHED_COMPONENT_NEW(component)	do {(void)(component);} while (0)
#define _STARPU_TRACE_SCHED_COMPONENT_CONNECT(parent, child)	do {(void)(parent); (void)(child);} while (0)
#define _STARPU_TRACE_SCHED_COMPONENT_PUSH(from, to, task, prio)	do {(void)(from); (void)(to); (void)(task); (void)(prio);} while (0)
#define _STARPU_TRACE_SCHED_COMPONENT_PULL(from, to, task)	do {(void)(from); (void)(to); (void)(task);} while (0)
#define _STARPU_TRACE_HANDLE_DATA_REGISTER(handle)	do {(void)(handle);} while (0)
#define _STARPU_TRACE_HANDLE_DATA_UNREGISTER(handle)	do {(void)(handle);} while (0)
#define _STARPU_TRACE_WORKER_START_FETCH_INPUT(job, id)	do {(void)(job); (void)(id);} while(0)
#define _STARPU_TRACE_WORKER_END_FETCH_INPUT(job, id)	do {(void)(job); (void)(id);} while(0)
#define _STARPU_TRACE_DATA_STATE_INVALID(handle, node)	do {(void)(handle); (void)(node);} while(0)
#define _STARPU_TRACE_DATA_STATE_OWNER(handle, node)	do {(void)(handle); (void)(node);} while(0)
#define _STARPU_TRACE_DATA_STATE_SHARED(handle, node)	do {(void)(handle); (void)(node);} while(0)
#define _STARPU_TRACE_DATA_REQUEST_CREATED(handle, orig, dest, prio, is_pre, req) do {(void)(handle); (void)(orig); (void)(dest); (void)(prio); (void)(is_pre); (void)(req); } while(0)
#define _STARPU_TRACE_PAPI_TASK_EVENT(event_id, task, value) do {(void)(event_id); (void)(task); (void)(value);} while(0)

#ifdef STARPU_BUBBLE
#define _STARPU_TRACE_BUBBLE_TASK_DEPS(a, b)	do {(void)(a); (void)(b);} while(0)
#define _STARPU_TRACE_BUBBLE(a)			do {(void)(a);} while(0)
#endif

#endif // STARPU_USE_FXT

#pragma GCC visibility pop

#endif // __FXT_H__
