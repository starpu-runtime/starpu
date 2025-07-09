/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018-2020  Federal University of Rio Grande do Sul (UFRGS)
 * Copyright (C) 2013-2013  Joris Pablo
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

#ifdef STARPU_RECURSIVE_TASKS
#define _STARPU_FUT_RECURSIVE_TASK		0x5189
#endif

#define _STARPU_FUT_TASK_RECURSIVE_SUBMIT		0x518c

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
 * - also adapt _starpu_profile_get_user_keymask() in src/profiling/fxt/fxt.c
 * - adapt KEYMASKALL_DEFAULT in src/profiling/fxt/fxt.c
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








/* TODO: the following macros are never called
 * -> shall we remove them ?
 */ 

#endif // STARPU_USE_FXT

#pragma GCC visibility pop

#endif // __FXT_H__
