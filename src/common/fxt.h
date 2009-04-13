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
#include <sys/syscall.h> /* pour les définitions de SYS_xxx */

#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <common/config.h>
#include <starpu.h>

/* some key to identify the worker kind */
#define FUT_APPS_KEY	0x100
#define FUT_CORE_KEY	0x101
#define FUT_CUDA_KEY	0x102

#define	FUT_NEW_WORKER_KEY	0x5102
#define	FUT_START_CODELET_BODY	0x5103
#define	FUT_END_CODELET_BODY	0x5104

#define FUT_JOB_PUSH		0x5105
#define FUT_JOB_POP		0x5106

#define FUT_START_FETCH_INPUT	0x5107
#define FUT_END_FETCH_INPUT	0x5108
#define FUT_START_PUSH_OUTPUT	0x5109
#define FUT_END_PUSH_OUTPUT	0x5110

#define FUT_CODELET_TAG		0x5111
#define FUT_CODELET_TAG_DEPS	0x5112

#define FUT_DATA_COPY		0x5113
#define FUT_WORK_STEALING	0x5114

#define	FUT_WORKER_TERMINATED	0x5115

#define FUT_USER_DEFINED_START	0x5116
#define FUT_USER_DEFINED_END	0x5117

#define	FUT_NEW_MEM_NODE	0x5118

#define	FUT_START_CALLBACK	0x5119
#define	FUT_END_CALLBACK	0x5120

#define	FUT_TASK_DONE		0x5121

#define	FUT_START_ALLOC		0x5122
#define	FUT_END_ALLOC		0x5123

#define	FUT_START_ALLOC_REUSE	0x5128
#define	FUT_END_ALLOC_REUSE	0x5129

#define	FUT_START_MEMRECLAIM	0x5124
#define	FUT_END_MEMRECLAIM	0x5125

#define	FUT_START_DRIVER_COPY	0x5126
#define	FUT_END_DRIVER_COPY	0x5127


#ifdef USE_FXT
#include <fxt/fxt.h>
#include <fxt/fut.h>

void start_fxt_profiling(void);
void fxt_register_thread(unsigned);

/* workerkind = FUT_CORE_KEY for instance */
#define TRACE_NEW_MEM_NODE(nodeid)	\
	FUT_DO_PROBE2(FUT_NEW_MEM_NODE, nodeid, syscall(SYS_gettid));

#define TRACE_NEW_WORKER(workerkind,memnode)	\
	FUT_DO_PROBE3(FUT_NEW_WORKER_KEY, workerkind, memnode, syscall(SYS_gettid));

#define TRACE_START_CODELET_BODY(job)	\
	FUT_DO_PROBE2(FUT_START_CODELET_BODY, job, syscall(SYS_gettid));

#define TRACE_END_CODELET_BODY(job)	\
	FUT_DO_PROBE2(FUT_END_CODELET_BODY, job, syscall(SYS_gettid));

#define TRACE_START_CALLBACK(job)	\
	FUT_DO_PROBE2(FUT_START_CALLBACK, job, syscall(SYS_gettid));

#define TRACE_END_CALLBACK(job)	\
	FUT_DO_PROBE2(FUT_END_CALLBACK, job, syscall(SYS_gettid));

#define TRACE_JOB_PUSH(task, prio)	\
	FUT_DO_PROBE3(FUT_JOB_PUSH, task, prio, syscall(SYS_gettid));

#define TRACE_JOB_POP(task, prio)	\
	FUT_DO_PROBE3(FUT_JOB_POP, task, prio, syscall(SYS_gettid));

#define TRACE_START_FETCH_INPUT(job)	\
	FUT_DO_PROBE2(FUT_START_FETCH_INPUT, job, syscall(SYS_gettid));

#define TRACE_END_FETCH_INPUT(job)	\
	FUT_DO_PROBE2(FUT_END_FETCH_INPUT, job, syscall(SYS_gettid));

#define TRACE_START_PUSH_OUTPUT(job)	\
	FUT_DO_PROBE2(FUT_START_PUSH_OUTPUT, job, syscall(SYS_gettid));

#define TRACE_END_PUSH_OUTPUT(job)	\
	FUT_DO_PROBE2(FUT_END_PUSH_OUTPUT, job, syscall(SYS_gettid));

#define TRACE_CODELET_TAG(tag, job)	\
	FUT_DO_PROBE2(FUT_CODELET_TAG, tag, job)

#define TRACE_CODELET_TAG_DEPS(tag_child, tag_father)	\
	FUT_DO_PROBE2(FUT_CODELET_TAG_DEPS, tag_child, tag_father)

#define TRACE_TASK_DONE(tag)	\
	FUT_DO_PROBE2(FUT_TASK_DONE, tag, syscall(SYS_gettid))

#define TRACE_DATA_COPY(src_node, dst_node, size)	\
	FUT_DO_PROBE3(FUT_DATA_COPY, src_node, dst_node, size)

#define TRACE_START_DRIVER_COPY(src_node, dst_node, size, com_id)	\
	FUT_DO_PROBE4(FUT_START_DRIVER_COPY, src_node, dst_node, size, com_id)

#define TRACE_END_DRIVER_COPY(src_node, dst_node, size, com_id)	\
	FUT_DO_PROBE4(FUT_END_DRIVER_COPY, src_node, dst_node, size, com_id)

#define TRACE_WORK_STEALING(empty_q, victim_q)		\
	FUT_DO_PROBE2(FUT_WORK_STEALING, empty_q, victim_q)

#define TRACE_WORKER_TERMINATED(workerkind)	\
	FUT_DO_PROBE2(FUT_WORKER_TERMINATED, workerkind, syscall(SYS_gettid));

#define TRACE_USER_DEFINED_START	\
	FUT_DO_PROBE1(FUT_USER_DEFINED_START, syscall(SYS_gettid));

#define TRACE_USER_DEFINED_END		\
	FUT_DO_PROBE1(FUT_USER_DEFINED_END, syscall(SYS_gettid));

#define TRACE_START_ALLOC(memnode)		\
	FUT_DO_PROBE2(FUT_START_ALLOC, memnode, syscall(SYS_gettid));
	
#define TRACE_END_ALLOC(memnode)		\
	FUT_DO_PROBE2(FUT_END_ALLOC, memnode, syscall(SYS_gettid));

#define TRACE_START_ALLOC_REUSE(memnode)		\
	FUT_DO_PROBE2(FUT_START_ALLOC_REUSE, memnode, syscall(SYS_gettid));
	
#define TRACE_END_ALLOC_REUSE(memnode)		\
	FUT_DO_PROBE2(FUT_END_ALLOC_REUSE, memnode, syscall(SYS_gettid));
	
#define TRACE_START_MEMRECLAIM(memnode)		\
	FUT_DO_PROBE2(FUT_START_MEMRECLAIM, memnode, syscall(SYS_gettid));
	
#define TRACE_END_MEMRECLAIM(memnode)		\
	FUT_DO_PROBE2(FUT_END_MEMRECLAIM, memnode, syscall(SYS_gettid));
	

#else // !USE_FXT

#define TRACE_NEW_MEM_NODE(nodeid)	do {} while(0);
#define TRACE_NEW_WORKER(a,b)		do {} while(0);
#define TRACE_START_CODELET_BODY(job)	do {} while(0);
#define TRACE_END_CODELET_BODY(job)	do {} while(0);
#define TRACE_START_CALLBACK(job)	do {} while(0);
#define TRACE_END_CALLBACK(job)		do {} while(0);
#define TRACE_JOB_PUSH(task, prio)	do {} while(0);
#define TRACE_JOB_POP(task, prio)	do {} while(0);
#define TRACE_START_FETCH_INPUT(job)	do {} while(0);
#define TRACE_END_FETCH_INPUT(job)	do {} while(0);
#define TRACE_START_PUSH_OUTPUT(job)	do {} while(0);
#define TRACE_END_PUSH_OUTPUT(job)	do {} while(0);
#define TRACE_CODELET_TAG(tag, job)	do {} while(0);
#define TRACE_CODELET_TAG_DEPS(a, b)	do {} while(0);
#define TRACE_TASK_DONE(tag)		do {} while(0);
#define TRACE_DATA_COPY(a, b, c)	do {} while(0);
#define TRACE_START_DRIVER_COPY(a,b,c,d)	do {} while(0);
#define TRACE_END_DRIVER_COPY(a,b,c,d)	do {} while(0);
#define TRACE_WORK_STEALING(a, b)	do {} while(0);
#define TRACE_WORKER_TERMINATED(a)	do {} while(0);
#define TRACE_USER_DEFINED_START	do {} while(0);
#define TRACE_USER_DEFINED_END		do {} while(0);
#define TRACE_START_ALLOC(memnode)	do {} while(0);
#define TRACE_END_ALLOC(memnode)	do {} while(0);
#define TRACE_START_ALLOC_REUSE(a)	do {} while(0);
#define TRACE_END_ALLOC_REUSE(a)	do {} while(0);
#define TRACE_START_MEMRECLAIM(memnode)	do {} while(0);
#define TRACE_END_MEMRECLAIM(memnode)	do {} while(0);

#endif // USE_FXT

#endif // __FXT_H__
