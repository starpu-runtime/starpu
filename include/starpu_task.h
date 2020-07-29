/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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
#include <starpu_util.h>
#include <starpu_task_bundle.h>
#include <errno.h>
#include <assert.h>

#if defined STARPU_USE_CUDA && !defined STARPU_DONT_INCLUDE_CUDA_HEADERS
# include <cuda.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define STARPU_NOWHERE	((1ULL)<<0)
#define STARPU_CPU	((1ULL)<<1)
#define STARPU_CUDA	((1ULL)<<3)
#define STARPU_OPENCL	((1ULL)<<6)
#define STARPU_MIC	((1ULL)<<7)
#define STARPU_SCC	((1ULL)<<8)

#define STARPU_CODELET_SIMGRID_EXECUTE	(1<<0)
#define STARPU_CUDA_ASYNC	(1<<0)
#define STARPU_OPENCL_ASYNC	(1<<0)

enum starpu_codelet_type
{
	STARPU_SEQ = 0,
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
	STARPU_TASK_BLOCKED_ON_DATA,
	STARPU_TASK_STOPPED
};

typedef uint64_t starpu_tag_t;

typedef void (*starpu_cpu_func_t)(void **, void*);
typedef void (*starpu_cuda_func_t)(void **, void*);
typedef void (*starpu_opencl_func_t)(void **, void*);
typedef void (*starpu_mic_kernel_t)(void **, void*);
typedef void (*starpu_scc_kernel_t)(void **, void*);

typedef starpu_mic_kernel_t (*starpu_mic_func_t)(void);
typedef starpu_scc_kernel_t (*starpu_scc_func_t)(void);

#define STARPU_MULTIPLE_CPU_IMPLEMENTATIONS    ((starpu_cpu_func_t) -1)
#define STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS   ((starpu_cuda_func_t) -1)
#define STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS ((starpu_opencl_func_t) -1)

#define STARPU_VARIABLE_NBUFFERS (-1)

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
	char cuda_flags[STARPU_MAXIMPLEMENTATIONS];
	starpu_opencl_func_t opencl_funcs[STARPU_MAXIMPLEMENTATIONS];
	char opencl_flags[STARPU_MAXIMPLEMENTATIONS];
	starpu_mic_func_t mic_funcs[STARPU_MAXIMPLEMENTATIONS];
	starpu_scc_func_t scc_funcs[STARPU_MAXIMPLEMENTATIONS];

	const char *cpu_funcs_name[STARPU_MAXIMPLEMENTATIONS];

	int nbuffers;
	enum starpu_data_access_mode modes[STARPU_NMAXBUFS];
	enum starpu_data_access_mode *dyn_modes;

	unsigned specific_nodes;
	int nodes[STARPU_NMAXBUFS];
	int *dyn_nodes;

	struct starpu_perfmodel *model;
	struct starpu_perfmodel *energy_model;

	unsigned long per_worker_stats[STARPU_NMAXWORKERS];

	const char *name;
	unsigned color;

	int flags;
};

struct starpu_task
{
	const char *name;

	struct starpu_codelet *cl;

	int nbuffers;

	starpu_data_handle_t handles[STARPU_NMAXBUFS];
	void *interfaces[STARPU_NMAXBUFS];
	enum starpu_data_access_mode modes[STARPU_NMAXBUFS];

	starpu_data_handle_t *dyn_handles;
	void **dyn_interfaces;
	enum starpu_data_access_mode *dyn_modes;

	void *cl_arg;
	size_t cl_arg_size;

	void (*callback_func)(void *);
	void *callback_arg;
	/* must StarPU release callback_arg ? - 0 by default */

	void (*prologue_callback_func)(void *);
	void *prologue_callback_arg;

	void (*prologue_callback_pop_func)(void *);
	void *prologue_callback_pop_arg;

	starpu_tag_t tag_id;

	unsigned cl_arg_free:1;
	unsigned callback_arg_free:1;
	/* must StarPU release prologue_callback_arg ? - 0 by default */
	unsigned prologue_callback_arg_free:1;
	/* must StarPU release prologue_callback_pop_arg ? - 0 by default */
	unsigned prologue_callback_pop_arg_free:1;

	unsigned use_tag:1;
	unsigned sequential_consistency:1;
	unsigned synchronous:1;
	unsigned execute_on_a_specific_worker:1;

	unsigned detach:1;
	unsigned destroy:1;
	unsigned regenerate:1;

	unsigned int mf_skip:1;

	unsigned scheduled:1;
	unsigned prefetched:1;

	unsigned workerid;
	unsigned workerorder;

	int priority;

	enum starpu_task_status status;

	int magic;
	unsigned type;
	unsigned color;

	unsigned sched_ctx;
	int hypervisor_tag;
	unsigned possibly_parallel;

	starpu_task_bundle_t bundle;

	struct starpu_profiling_task_info *profiling_info;

	double flops;
	double predicted;
	double predicted_transfer;

	struct starpu_task *prev;
	struct starpu_task *next;
	void *starpu_private;
#ifdef STARPU_OPENMP
	struct starpu_omp_task *omp_task;
#else
	void *omp_task;
#endif
};

#define STARPU_TASK_TYPE_NORMAL		0
#define STARPU_TASK_TYPE_INTERNAL	(1<<0)
#define STARPU_TASK_TYPE_DATA_ACQUIRE	(1<<1)

/* Note: remember to update starpu_task_init as well */
#define STARPU_TASK_INITIALIZER 			\
{							\
	.cl = NULL,					\
	.cl_arg = NULL,					\
	.cl_arg_size = 0,				\
	.callback_func = NULL,				\
	.callback_arg = NULL,				\
	.priority = STARPU_DEFAULT_PRIO,		\
	.use_tag = 0,					\
	.sequential_consistency = 1,			\
	.synchronous = 0,				\
	.execute_on_a_specific_worker = 0,		\
	.workerorder = 0,				\
	.bundle = NULL,					\
	.detach = 1,					\
	.destroy = 0,					\
	.regenerate = 0,				\
	.status = STARPU_TASK_INVALID,			\
	.profiling_info = NULL,				\
	.predicted = NAN,				\
	.predicted_transfer = NAN,			\
	.starpu_private = NULL,				\
	.magic = 42,                  			\
	.type = 0,					\
	.color = 0,					\
	.sched_ctx = STARPU_NMAX_SCHED_CTXS,		\
	.hypervisor_tag = 0,				\
	.flops = 0.0,					\
	.scheduled = 0,					\
	.prefetched = 0,				\
	.dyn_handles = NULL,				\
	.dyn_interfaces = NULL,				\
	.dyn_modes = NULL,				\
	.name = NULL,                        		\
	.possibly_parallel = 0                        	\
}

#define STARPU_TASK_GET_NBUFFERS(task) ((unsigned)((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS ? ((task)->nbuffers) : ((task)->cl->nbuffers)))

#define STARPU_TASK_GET_HANDLE(task, i) (((task)->dyn_handles) ? (task)->dyn_handles[i] : (task)->handles[i])
#define STARPU_TASK_GET_HANDLES(task) (((task)->dyn_handles) ? (task)->dyn_handles : (task)->handles)
#define STARPU_TASK_SET_HANDLE(task, handle, i) do { if ((task)->dyn_handles) (task)->dyn_handles[i] = handle; else (task)->handles[i] = handle; } while(0)

#define STARPU_CODELET_GET_MODE(codelet, i) (((codelet)->dyn_modes) ? (codelet)->dyn_modes[i] : (assert(i < STARPU_NMAXBUFS), (codelet)->modes[i]))
#define STARPU_CODELET_SET_MODE(codelet, mode, i) do { if ((codelet)->dyn_modes) (codelet)->dyn_modes[i] = mode; else (codelet)->modes[i] = mode; } while(0)

#define STARPU_TASK_GET_MODE(task, i) ((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (task)->dyn_modes ? \
						(((task)->dyn_modes) ? (task)->dyn_modes[i] : (task)->modes[i]) : \
						STARPU_CODELET_GET_MODE((task)->cl, i) )
#define STARPU_TASK_SET_MODE(task, mode, i) do { \
					if ((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (task)->cl->nbuffers > STARPU_NMAXBUFS) \
						if ((task)->dyn_modes) (task)->dyn_modes[i] = mode; else (task)->modes[i] = mode; \
					else \
						STARPU_CODELET_SET_MODE((task)->cl, mode, i); \
					} while(0)

#define STARPU_CODELET_GET_NODE(codelet, i) (((codelet)->dyn_nodes) ? (codelet)->dyn_nodes[i] : (codelet)->nodes[i])
#define STARPU_CODELET_SET_NODE(codelet, __node, i) do { if ((codelet)->dyn_nodes) (codelet)->dyn_nodes[i] = __node; else (codelet)->nodes[i] = __node; } while(0)

void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...);
void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array);

void starpu_task_declare_deps_array(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);

int starpu_task_get_task_succs(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);
int starpu_task_get_task_scheduled_succs(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);

int starpu_tag_wait(starpu_tag_t id);
int starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id);

void starpu_tag_notify_from_apps(starpu_tag_t id);

void starpu_tag_restart(starpu_tag_t id);

void starpu_tag_notify_restart_from_apps(starpu_tag_t id);

void starpu_tag_remove(starpu_tag_t id);

void starpu_task_init(struct starpu_task *task);
void starpu_task_clean(struct starpu_task *task);

struct starpu_task *starpu_task_create(void) STARPU_ATTRIBUTE_MALLOC;

void starpu_task_destroy(struct starpu_task *task);
int starpu_task_submit(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;
int starpu_task_submit_to_ctx(struct starpu_task *task, unsigned sched_ctx_id);

int starpu_task_finished(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

int starpu_task_wait(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

int starpu_task_wait_for_all(void);
int starpu_task_wait_for_n_submitted(unsigned n);

int starpu_task_wait_for_all_in_ctx(unsigned sched_ctx_id);
int starpu_task_wait_for_n_submitted_in_ctx(unsigned sched_ctx_id, unsigned n);

int starpu_task_wait_for_no_ready(void);

int starpu_task_nready(void);
int starpu_task_nsubmitted(void);

void starpu_iteration_push(unsigned long iteration);
void starpu_iteration_pop(void);

void starpu_do_schedule(void);

void starpu_codelet_init(struct starpu_codelet *cl);

void starpu_codelet_display_stats(struct starpu_codelet *cl);

struct starpu_task *starpu_task_get_current(void);

const char *starpu_task_get_model_name(struct starpu_task *task);
const char *starpu_task_get_name(struct starpu_task *task);

void starpu_parallel_task_barrier_init(struct starpu_task *task, int workerid);
void starpu_parallel_task_barrier_init_n(struct starpu_task *task, int worker_size);

struct starpu_task *starpu_task_dup(struct starpu_task *task);

void starpu_task_set_implementation(struct starpu_task *task, unsigned impl);
unsigned starpu_task_get_implementation(struct starpu_task *task);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_H__ */
