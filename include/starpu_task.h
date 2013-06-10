/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2011  INRIA
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

#ifndef __STARPU_TASK_H__
#define __STARPU_TASK_H__

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_task_bundle.h>
#include <errno.h>

#if defined STARPU_USE_CUDA && !defined STARPU_DONT_INCLUDE_CUDA_HEADERS
# include <cuda.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define STARPU_CPU	((1ULL)<<1)
#define STARPU_CUDA	((1ULL)<<3)
#define STARPU_OPENCL	((1ULL)<<6)

enum starpu_codelet_type
{
	STARPU_SEQ,
	STARPU_SPMD,
	STARPU_FORKJOIN
};

enum starpu_task_status
{
	STARPU_TASK_INVALID,
#define STARPU_TASK_INVALID 0
	STARPU_TASK_BLOCKED,
	STARPU_TASK_READY,
	STARPU_TASK_RUNNING,
	STARPU_TASK_FINISHED,
	STARPU_TASK_BLOCKED_ON_TAG,
	STARPU_TASK_BLOCKED_ON_TASK,
	STARPU_TASK_BLOCKED_ON_DATA
};

typedef uint64_t starpu_tag_t;

typedef void (*starpu_cpu_func_t)(void **, void*);
typedef void (*starpu_cuda_func_t)(void **, void*);
typedef void (*starpu_opencl_func_t)(void **, void*);

#define STARPU_MULTIPLE_CPU_IMPLEMENTATIONS    ((starpu_cpu_func_t) -1)
#define STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS   ((starpu_cuda_func_t) -1)
#define STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS ((starpu_opencl_func_t) -1)

struct starpu_task;
struct starpu_codelet
{
	uint32_t where;
	int (*can_execute)(unsigned workerid, struct starpu_task *task, unsigned nimpl);
	enum starpu_codelet_type type;
	int max_parallelism;

	starpu_cpu_func_t cpu_func STARPU_DEPRECATED;
	starpu_cuda_func_t cuda_func STARPU_DEPRECATED;
	starpu_opencl_func_t opencl_func STARPU_DEPRECATED;

	starpu_cpu_func_t cpu_funcs[STARPU_MAXIMPLEMENTATIONS];
	starpu_cuda_func_t cuda_funcs[STARPU_MAXIMPLEMENTATIONS];
	starpu_opencl_func_t opencl_funcs[STARPU_MAXIMPLEMENTATIONS];

	unsigned nbuffers;
	enum starpu_data_access_mode modes[STARPU_NMAXBUFS];
	enum starpu_data_access_mode *dyn_modes;

	struct starpu_perfmodel *model;
	struct starpu_perfmodel *power_model;

	unsigned long per_worker_stats[STARPU_NMAXWORKERS];

	const char *name;
};

struct starpu_task
{
	struct starpu_codelet *cl;

	struct starpu_data_descr buffers[STARPU_NMAXBUFS] STARPU_DEPRECATED;
	starpu_data_handle_t handles[STARPU_NMAXBUFS];
	void *interfaces[STARPU_NMAXBUFS];

	starpu_data_handle_t *dyn_handles;
	void **dyn_interfaces;

	void *cl_arg;
	size_t cl_arg_size;

	void (*callback_func)(void *);
	void *callback_arg;

	unsigned use_tag;
	starpu_tag_t tag_id;

	unsigned sequential_consistency;

	unsigned synchronous;
	int priority;

	unsigned execute_on_a_specific_worker;
	unsigned workerid;

	starpu_task_bundle_t bundle;

	int detach;
	int destroy;
	int regenerate;

	enum starpu_task_status status;

	struct starpu_profiling_task_info *profiling_info;

	double predicted;
	double predicted_transfer;

	unsigned int mf_skip;

	struct starpu_task *prev;
	struct starpu_task *next;
	void *starpu_private;
	int magic;

	unsigned sched_ctx;
	int hypervisor_tag;
	double flops;

	unsigned scheduled;
};

#define STARPU_TASK_INITIALIZER 			\
{							\
	.cl = NULL,					\
	.cl_arg = NULL,					\
	.cl_arg_size = 0,				\
	.callback_func = NULL,				\
	.callback_arg = NULL,				\
	.priority = STARPU_DEFAULT_PRIO,		\
	.use_tag = 0,					\
	.synchronous = 0,				\
	.execute_on_a_specific_worker = 0,		\
	.bundle = NULL,					\
	.detach = 1,					\
	.destroy = 0,					\
	.regenerate = 0,				\
	.status = STARPU_TASK_INVALID,			\
	.profiling_info = NULL,				\
	.predicted = -1.0,				\
	.predicted_transfer = -1.0,			\
	.starpu_private = NULL,				\
	.magic = 42,                  			\
	.sched_ctx = 0,					\
	.hypervisor_tag = 0,				\
	.flops = 0.0,					\
	.scheduled = 0,					\
	.dyn_handles = NULL,				\
	.dyn_interfaces = NULL				\
}

#define STARPU_TASK_GET_HANDLE(task, i) ((task->dyn_handles) ? task->dyn_handles[i] : task->handles[i])
#define STARPU_TASK_SET_HANDLE(task, handle, i) do { if (task->dyn_handles) task->dyn_handles[i] = handle; else task->handles[i] = handle; } while(0)

#define STARPU_CODELET_GET_MODE(codelet, i) ((codelet->dyn_modes) ? codelet->dyn_modes[i] : codelet->modes[i])
#define STARPU_CODELET_SET_MODE(codelet, mode, i) do { if (codelet->dyn_modes) codelet->dyn_modes[i] = mode; else codelet->modes[i] = mode; } while(0)

void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...);
void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array);

void starpu_task_declare_deps_array(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);

int starpu_tag_wait(starpu_tag_t id);
int starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id);

void starpu_tag_notify_from_apps(starpu_tag_t id);

void starpu_tag_restart(starpu_tag_t id);

void starpu_tag_remove(starpu_tag_t id);

void starpu_task_init(struct starpu_task *task);
void starpu_task_clean(struct starpu_task *task);

struct starpu_task *starpu_task_create(void);

void starpu_task_destroy(struct starpu_task *task);
int starpu_task_submit(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;
int starpu_task_submit_to_ctx(struct starpu_task *task, unsigned sched_ctx_id);

int starpu_task_wait(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

int starpu_task_wait_for_all(void);

int starpu_task_wait_for_all_in_ctx(unsigned sched_ctx_id);

int starpu_task_wait_for_no_ready(void);

int starpu_task_nready(void);
int starpu_task_nsubmitted(void);

void starpu_codelet_init(struct starpu_codelet *cl);

void starpu_codelet_display_stats(struct starpu_codelet *cl);

struct starpu_task *starpu_task_get_current(void);

void starpu_parallel_task_barrier_init(struct starpu_task* task, int workerid);

struct starpu_task *starpu_task_dup(struct starpu_task *task);

void starpu_task_set_implementation(struct starpu_task *task, unsigned impl);
unsigned starpu_task_get_implementation(struct starpu_task *task);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_H__ */
