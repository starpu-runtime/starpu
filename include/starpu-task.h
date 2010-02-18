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

#ifndef __STARPU_TASK_H__
#define __STARPU_TASK_H__

#include <errno.h>
#include <starpu_config.h>
#include <starpu.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif

#include <starpu-data.h>

#define STARPU_CPU	((1ULL)<<1)
#define STARPU_CUDA	((1ULL)<<3)
#define STARPU_SPU	((1ULL)<<4)
#define STARPU_GORDON	((1ULL)<<5)

#define STARPU_MIN_PRIO        (-4)
#define STARPU_MAX_PRIO        5
#define STARPU_DEFAULT_PRIO	0

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t starpu_tag_t;

/*
 * A codelet describes the various function 
 * that may be called from a worker
 */
typedef struct starpu_codelet_t {
	/* where can it be performed ? */
	uint32_t where;

	/* the different implementations of the codelet */
	void (*cuda_func)(void **, void *);
	void (*cpu_func)(void **, void *);
	uint8_t gordon_func;

	/* how many buffers do the codelet takes as argument ? */
	unsigned nbuffers;

	struct starpu_perfmodel_t *model;

	/* statistics collected at runtime: this is filled by StarPU and should
	 * not be accessed directly (use the starpu_display_codelet_stats
	 * function instead for instance). */
	unsigned long per_worker_stats[STARPU_NMAXWORKERS];
} starpu_codelet;

struct starpu_task {
	struct starpu_codelet_t *cl;

	/* arguments managed by the DSM */
	struct starpu_buffer_descr_t buffers[STARPU_NMAXBUFS];
	void *interface[STARPU_NMAXBUFS];

	/* arguments not managed by the DSM are given as a buffer */
	void *cl_arg;
	/* in case the argument buffer has to be uploaded explicitely */
	size_t cl_arg_size;
	
	/* when the task is done, callback_func(callback_arg) is called */
	void (*callback_func)(void *);
	void *callback_arg;

	unsigned use_tag;
	starpu_tag_t tag_id;

	/* options for the task execution */
	unsigned synchronous; /* if set, a call to push is blocking */
	int priority; /* STARPU_MAX_PRIO = most important 
        		: STARPU_MIN_PRIO = least important */

	/* in case the task has to be executed on a specific worker */
	unsigned execute_on_a_specific_worker;
	unsigned workerid;

	/* If this flag is set, it is not possible to synchronize with the task
	 * by the means of starpu_wait_task later on. Internal data structures
	 * are only garanteed to be liberated once starpu_wait_task is called
	 * if that flag is not set. */
	int detach;

	/* If that flag is set, the task structure will automatically be
	 * liberated, either after the execution of the callback if the task is
	 * detached, or during starpu_task_wait otherwise. If this flag is not
	 * set, dynamically allocated data structures will not be liberated
	 * until starpu_task_destroy is called explicitely. Setting this flag
	 * for a statically allocated task structure will result in undefined
	 * behaviour. */
	int destroy;

	/* If this flag is set, the task will be re-submitted to StarPU once it
	 * has been executed. This flag must not be set if the destroy flag is
	 * set too. */ 
	int regenerate;

	/* this is private to StarPU, do not modify. If the task is allocated
	 * by hand (without starpu_task_create), this field should be set to
	 * NULL. */
	void *starpu_private;
};

/* It is possible to initialize statically allocated tasks with this value.
 * This is equivalent to initializing a starpu_task structure with the
 * starpu_task_init function. */
#define STARPU_TASK_INITIALIZER 			\
{							\
	.cl = NULL,					\
	.cl_arg = NULL,					\
	.cl_arg_size = 0,				\
	.callback_func = NULL,				\
	.callback_arg = NULL,				\
	.priority = STARPU_DEFAULT_PRIO,                \
	.use_tag = 0,					\
	.synchronous = 0,				\
	.execute_on_a_specific_worker = 0,		\
	.detach = 1,					\
	.destroy = 0,					\
	.regenerate = 0,				\
	.starpu_private = NULL				\
};

/*
 * handle task dependencies: it is possible to associate a task with a unique
 * "tag" and to express dependencies between tasks by the means of those tags
 *
 * To do so, fill the tag_id field with a tag number (can be arbitrary) and set
 * use_tag to 1.
 *
 * If starpu_tag_declare_deps is called with that tag number, the task will not
 * be started until the task which wears the declared dependency tags are
 * complete.
 */

/*
 * WARNING ! use with caution ...
 *  In case starpu_tag_declare_deps is passed constant arguments, the caller
 *  must make sure that the constants are casted to starpu_tag_t. Otherwise,
 *  due to integer sizes and argument passing on the stack, the C compiler
 *  might consider the tag *  0x200000003 instead of 0x2 and 0x3 when calling:
 *      "starpu_tag_declare_deps(0x1, 2, 0x2, 0x3)"
 *  Using starpu_tag_declare_deps_array is a way to avoid this problem.
 */
/* make id depend on the list of ids */
void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...);
void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array);

int starpu_tag_wait(starpu_tag_t id);
int starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id);

/* The application can feed a tag explicitely */
void starpu_tag_notify_from_apps(starpu_tag_t id);

/* To release resources, tags should be freed after use */
void starpu_tag_remove(starpu_tag_t id);

/* Initialize a task structure with default values. */
void starpu_task_init(struct starpu_task *task);

/* Allocate a task structure and initialize it with default values. Tasks
 * allocated dynamically with starpu_task_create are automatically liberated
 * when the task is terminated. If the destroy flag is explicitely unset, the
 * ressources used by the task are liberated by calling starpu_task_destroy.
 * */
struct starpu_task *starpu_task_create(void);

/* Liberate the ressource allocated during starpu_task_create. This function
 * can be called automatically after the execution of a task by setting the
 * "destroy" flag of the starpu_task structure (default behaviour). Calling
 * this function on a statically allocated task results in an undefined
 * behaviour. */
void starpu_task_destroy(struct starpu_task *task);
int starpu_submit_task(struct starpu_task *task);

/* This function blocks until the task was executed. It is not possible to
 * synchronize with a task more than once. It is not possible to wait
 * synchronous or detached tasks.
 * Upon successful completion, this function returns 0. Otherwise, -EINVAL
 * indicates that the waited task was either synchronous or detached. */
int starpu_wait_task(struct starpu_task *task);

/* This function waits until all the tasks that were already submitted have
 * been executed. */
int starpu_wait_all_tasks(void);

void starpu_display_codelet_stats(struct starpu_codelet_t *cl);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_TASK_H__
