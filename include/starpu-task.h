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

/* this is a randomly choosen value ... */
#ifndef MAXCUDADEVS
#define MAXCUDADEVS     4
#endif

#ifdef USE_CUDA
#include <cuda.h>
#endif

#include <starpu-data.h>

#define CORE	((1ULL)<<1)
#define CUBLAS	((1ULL)<<2)
#define CUDA	((1ULL)<<3)
#define SPU	((1ULL)<<4)
#define GORDON	((1ULL)<<5)

#define MIN_PRIO        (-4)
#define MAX_PRIO        5
#define DEFAULT_PRIO	0

typedef uint64_t starpu_tag_t;

/*
 * A codelet describes the various function 
 * that may be called from a worker
 */
typedef struct starpu_codelet_t {
	/* where can it be performed ? */
	uint32_t where;

	/* the different implementations of the codelet */
	void *cuda_func;
	void *core_func;
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
	struct starpu_buffer_descr_t buffers[NMAXBUFS];
	starpu_data_interface_t interface[NMAXBUFS];

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
	int priority; /* MAX_PRIO = most important 
        		: MIN_PRIO = least important */

	/* in case the task has to be executed on a specific worker */
	unsigned execute_on_a_specific_worker;
	unsigned workerid;

	/* should the task be automatically liberated once executed ? */
	int cleanup;

	/* this is private to StarPU, do not modify. If the task is allocated
	 * by hand (without starpu_task_create), this field should be set to
	 * NULL. */
	void *starpu_private;
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

void starpu_tag_wait(starpu_tag_t id);
void starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id);

/* The application can feed a tag explicitely */
void starpu_tag_notify_from_apps(starpu_tag_t id);

/* To release resources, tags should be freed after use */
void starpu_tag_remove(starpu_tag_t id);

struct starpu_task *starpu_task_create(void);
int starpu_submit_task(struct starpu_task *task);

void starpu_display_codelet_stats(struct starpu_codelet_t *cl);

#endif // __STARPU_TASK_H__
