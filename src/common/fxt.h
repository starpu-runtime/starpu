/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __FXT_H__
#define __FXT_H__


#ifndef _GNU_SOURCE
#define _GNU_SOURCE  /* ou _BSD_SOURCE ou _SVID_SOURCE */
#endif

#include <unistd.h>

#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <common/config.h>
#include <starpu.h>

/* some key to identify the worker kind */
#define STARPU_FUT_APPS_KEY	0x100
#define STARPU_FUT_CPU_KEY	0x101
#define STARPU_FUT_CUDA_KEY	0x102

#define STARPU_FUT_WORKER_INIT_START	0x5133
#define STARPU_FUT_WORKER_INIT_END	0x5134

#define	STARPU_FUT_START_CODELET_BODY	0x5103
#define	STARPU_FUT_END_CODELET_BODY	0x5104

#define STARPU_FUT_JOB_PUSH		0x5105
#define STARPU_FUT_JOB_POP		0x5106

#define STARPU_FUT_START_FETCH_INPUT	0x5107
#define STARPU_FUT_END_FETCH_INPUT	0x5108
#define STARPU_FUT_START_PUSH_OUTPUT	0x5109
#define STARPU_FUT_END_PUSH_OUTPUT	0x5110

#define STARPU_FUT_CODELET_TAG		0x5111
#define STARPU_FUT_CODELET_TAG_DEPS	0x5112

#define STARPU_FUT_DATA_COPY		0x5113
#define STARPU_FUT_WORK_STEALING	0x5114

#define STARPU_FUT_WORKER_DEINIT_START	0x5135
#define STARPU_FUT_WORKER_DEINIT_END	0x5136

#define STARPU_FUT_USER_DEFINED_START	0x5116
#define STARPU_FUT_USER_DEFINED_END	0x5117

#define	STARPU_FUT_NEW_MEM_NODE	0x5118

#define	STARPU_FUT_START_CALLBACK	0x5119
#define	STARPU_FUT_END_CALLBACK	0x5120

#define	STARPU_FUT_TASK_DONE		0x5121

#define	STARPU_FUT_START_ALLOC		0x5122
#define	STARPU_FUT_END_ALLOC		0x5123

#define	STARPU_FUT_START_ALLOC_REUSE	0x5128
#define	STARPU_FUT_END_ALLOC_REUSE	0x5129

#define	STARPU_FUT_START_MEMRECLAIM	0x5124
#define	STARPU_FUT_END_MEMRECLAIM	0x5125

#define	STARPU_FUT_START_DRIVER_COPY	0x5126
#define	STARPU_FUT_END_DRIVER_COPY	0x5127

#define	STARPU_FUT_START_PROGRESS	0x5130
#define	STARPU_FUT_END_PROGRESS	0x5131

#define STARPU_FUT_USER_EVENT		0x5132

#ifdef STARPU_USE_FXT
#include <sys/syscall.h> /* pour les définitions de SYS_xxx */
#include <fxt/fxt.h>
#include <fxt/fut.h>

void _starpu_start_fxt_profiling(void);
void _starpu_stop_fxt_profiling(void);
void _starpu_fxt_register_thread(unsigned);

/* sometimes we need something a little more specific than the wrappers from
 * FxT */
#define STARPU_FUT_DO_PROBE3STR(CODE, P1, P2, P3, str)				\
do {									\
	/* we add a \0 just in case ... */				\
	size_t len = strlen((str)) + 1;					\
	unsigned nbargs = 3 + (len + sizeof(unsigned long) - 1)/(sizeof(unsigned long));\
	size_t total_len = FUT_SIZE(nbargs);				\
	unsigned long *args =						\
		fut_getstampedbuffer(FUT_CODE(CODE, nbargs), total_len);\
	*(args++) = (unsigned long)(P1);				\
	*(args++) = (unsigned long)(P2);				\
	*(args++) = (unsigned long)(P3);				\
	sprintf((char *)args, "%s", str);				\
} while (0);

/* workerkind = STARPU_FUT_CPU_KEY for instance */
#define STARPU_TRACE_NEW_MEM_NODE(nodeid)			\
	FUT_DO_PROBE2(STARPU_FUT_NEW_MEM_NODE, nodeid, syscall(SYS_gettid));

#define STARPU_TRACE_WORKER_INIT_START(workerkind,memnode)	\
	FUT_DO_PROBE3(STARPU_FUT_WORKER_INIT_START, workerkind, memnode, syscall(SYS_gettid));

#define STARPU_TRACE_WORKER_INIT_END				\
	FUT_DO_PROBE1(STARPU_FUT_WORKER_INIT_END, syscall(SYS_gettid));

#define STARPU_TRACE_START_CODELET_BODY(job)					\
do {									\
	struct starpu_perfmodel_t *model = (job)->task->cl->model;	\
	if (model && model->symbol)					\
	{								\
		/* we include the symbol name */			\
		STARPU_FUT_DO_PROBE3STR(STARPU_FUT_START_CODELET_BODY, job, syscall(SYS_gettid), 1, model->symbol);\
	}								\
	else {								\
		FUT_DO_PROBE3(STARPU_FUT_START_CODELET_BODY, job, syscall(SYS_gettid), 0);\
	}								\
} while(0);


#define STARPU_TRACE_END_CODELET_BODY(job)	\
	FUT_DO_PROBE2(STARPU_FUT_END_CODELET_BODY, job, syscall(SYS_gettid));

#define STARPU_TRACE_START_CALLBACK(job)	\
	FUT_DO_PROBE2(STARPU_FUT_START_CALLBACK, job, syscall(SYS_gettid));

#define STARPU_TRACE_END_CALLBACK(job)	\
	FUT_DO_PROBE2(STARPU_FUT_END_CALLBACK, job, syscall(SYS_gettid));

#define STARPU_TRACE_JOB_PUSH(task, prio)	\
	FUT_DO_PROBE3(STARPU_FUT_JOB_PUSH, task, prio, syscall(SYS_gettid));

#define STARPU_TRACE_JOB_POP(task, prio)	\
	FUT_DO_PROBE3(STARPU_FUT_JOB_POP, task, prio, syscall(SYS_gettid));

#define STARPU_TRACE_START_FETCH_INPUT(job)	\
	FUT_DO_PROBE2(STARPU_FUT_START_FETCH_INPUT, job, syscall(SYS_gettid));

#define STARPU_TRACE_END_FETCH_INPUT(job)	\
	FUT_DO_PROBE2(STARPU_FUT_END_FETCH_INPUT, job, syscall(SYS_gettid));

#define STARPU_TRACE_START_PUSH_OUTPUT(job)	\
	FUT_DO_PROBE2(STARPU_FUT_START_PUSH_OUTPUT, job, syscall(SYS_gettid));

#define STARPU_TRACE_END_PUSH_OUTPUT(job)	\
	FUT_DO_PROBE2(STARPU_FUT_END_PUSH_OUTPUT, job, syscall(SYS_gettid));

#define STARPU_TRACE_CODELET_TAG(tag, job)	\
	FUT_DO_PROBE2(STARPU_FUT_CODELET_TAG, tag, job)

#define STARPU_TRACE_CODELET_TAG_DEPS(tag_child, tag_father)	\
	FUT_DO_PROBE2(STARPU_FUT_CODELET_TAG_DEPS, tag_child, tag_father)

#define STARPU_TRACE_TASK_DONE(tag)							\
do {										\
	struct starpu_job_s *job = (tag)->job;						\
	if (job && job->task 							\
		&& job->task->cl						\
		&& job->task->cl->model						\
		&& job->task->cl->model->symbol)				\
	{									\
		char *symbol = job->task->cl->model->symbol;			\
		STARPU_FUT_DO_PROBE3STR(STARPU_FUT_TASK_DONE, tag->id, syscall(SYS_gettid), 1, symbol);\
	}									\
	else {									\
		FUT_DO_PROBE3(STARPU_FUT_TASK_DONE, tag->id, syscall(SYS_gettid), 0);	\
	}									\
} while(0);

#define STARPU_TRACE_DATA_COPY(src_node, dst_node, size)	\
	FUT_DO_PROBE3(STARPU_FUT_DATA_COPY, src_node, dst_node, size)

#define STARPU_TRACE_START_DRIVER_COPY(src_node, dst_node, size, com_id)	\
	FUT_DO_PROBE4(STARPU_FUT_START_DRIVER_COPY, src_node, dst_node, size, com_id)

#define STARPU_TRACE_END_DRIVER_COPY(src_node, dst_node, size, com_id)	\
	FUT_DO_PROBE4(STARPU_FUT_END_DRIVER_COPY, src_node, dst_node, size, com_id)

#define STARPU_TRACE_WORK_STEALING(empty_q, victim_q)		\
	FUT_DO_PROBE2(STARPU_FUT_WORK_STEALING, empty_q, victim_q)

#define STARPU_TRACE_WORKER_DEINIT_START			\
	FUT_DO_PROBE1(STARPU_FUT_WORKER_DEINIT_START, syscall(SYS_gettid));

#define STARPU_TRACE_WORKER_DEINIT_END(workerkind)		\
	FUT_DO_PROBE2(STARPU_FUT_WORKER_DEINIT_END, workerkind, syscall(SYS_gettid));

#define STARPU_TRACE_USER_DEFINED_START	\
	FUT_DO_PROBE1(STARPU_FUT_USER_DEFINED_START, syscall(SYS_gettid));

#define STARPU_TRACE_USER_DEFINED_END		\
	FUT_DO_PROBE1(STARPU_FUT_USER_DEFINED_END, syscall(SYS_gettid));

#define STARPU_TRACE_START_ALLOC(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_START_ALLOC, memnode, syscall(SYS_gettid));
	
#define STARPU_TRACE_END_ALLOC(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_END_ALLOC, memnode, syscall(SYS_gettid));

#define STARPU_TRACE_START_ALLOC_REUSE(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_START_ALLOC_REUSE, memnode, syscall(SYS_gettid));
	
#define STARPU_TRACE_END_ALLOC_REUSE(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_END_ALLOC_REUSE, memnode, syscall(SYS_gettid));
	
#define STARPU_TRACE_START_MEMRECLAIM(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_START_MEMRECLAIM, memnode, syscall(SYS_gettid));
	
#define STARPU_TRACE_END_MEMRECLAIM(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_END_MEMRECLAIM, memnode, syscall(SYS_gettid));
	
#define STARPU_TRACE_START_PROGRESS(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_START_PROGRESS, memnode, syscall(SYS_gettid));

#define STARPU_TRACE_END_PROGRESS(memnode)		\
	FUT_DO_PROBE2(STARPU_FUT_END_PROGRESS, memnode, syscall(SYS_gettid));
	
#define STARPU_TRACE_USER_EVENT(code)			\
	FUT_DO_PROBE2(STARPU_FUT_USER_EVENT, code, syscall(SYS_gettid));

#else // !STARPU_USE_FXT

#define STARPU_TRACE_NEW_MEM_NODE(nodeid)	do {} while(0);
#define TRACE_NEW_WORKER(a,b)		do {} while(0);
#define STARPU_TRACE_WORKER_INIT_START(a,b)	do {} while(0);
#define STARPU_TRACE_WORKER_INIT_END		do {} while(0);
#define STARPU_TRACE_START_CODELET_BODY(job)	do {} while(0);
#define STARPU_TRACE_END_CODELET_BODY(job)	do {} while(0);
#define STARPU_TRACE_START_CALLBACK(job)	do {} while(0);
#define STARPU_TRACE_END_CALLBACK(job)		do {} while(0);
#define STARPU_TRACE_JOB_PUSH(task, prio)	do {} while(0);
#define STARPU_TRACE_JOB_POP(task, prio)	do {} while(0);
#define STARPU_TRACE_START_FETCH_INPUT(job)	do {} while(0);
#define STARPU_TRACE_END_FETCH_INPUT(job)	do {} while(0);
#define STARPU_TRACE_START_PUSH_OUTPUT(job)	do {} while(0);
#define STARPU_TRACE_END_PUSH_OUTPUT(job)	do {} while(0);
#define STARPU_TRACE_CODELET_TAG(tag, job)	do {} while(0);
#define STARPU_TRACE_CODELET_TAG_DEPS(a, b)	do {} while(0);
#define STARPU_TRACE_TASK_DONE(tag)		do {} while(0);
#define STARPU_TRACE_DATA_COPY(a, b, c)	do {} while(0);
#define STARPU_TRACE_START_DRIVER_COPY(a,b,c,d)	do {} while(0);
#define STARPU_TRACE_END_DRIVER_COPY(a,b,c,d)	do {} while(0);
#define STARPU_TRACE_WORK_STEALING(a, b)	do {} while(0);
#define STARPU_TRACE_WORKER_DEINIT_START	do {} while(0);
#define STARPU_TRACE_WORKER_DEINIT_END(a)	do {} while(0);
#define STARPU_TRACE_USER_DEFINED_START	do {} while(0);
#define STARPU_TRACE_USER_DEFINED_END		do {} while(0);
#define STARPU_TRACE_START_ALLOC(memnode)	do {} while(0);
#define STARPU_TRACE_END_ALLOC(memnode)	do {} while(0);
#define STARPU_TRACE_START_ALLOC_REUSE(a)	do {} while(0);
#define STARPU_TRACE_END_ALLOC_REUSE(a)	do {} while(0);
#define STARPU_TRACE_START_MEMRECLAIM(memnode)	do {} while(0);
#define STARPU_TRACE_END_MEMRECLAIM(memnode)	do {} while(0);
#define STARPU_TRACE_START_PROGRESS(memnode)	do {} while(0);
#define STARPU_TRACE_END_PROGRESS(memnode)	do {} while(0);
#define STARPU_TRACE_USER_EVENT(code)		do {} while(0);

#endif // STARPU_USE_FXT

#endif // __FXT_H__
