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

#ifndef __STARPU_H__
#define __STARPU_H__

#include <stdlib.h>
#include <stdint.h>

/* Maximum number of workers supported by StarPU, the actual number of worker
 * is given by the startpu_get_worker_count method */
#define STARPU_NMAXWORKERS	32

#include <starpu_config.h>
#include <starpu-util.h>
#include <starpu-data.h>
#include <starpu-perfmodel.h>
#include <starpu-task.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TODO: should either make 0 be the default, or provide an initializer, to
 * make future extensions not problematic */
struct starpu_conf {
	/* which scheduling policy should be used ? (NULL for default) */
	const char *sched_policy;

	/* maximum number of CPUs (-1 for default) */
	int ncpus;
	/* maximum number of CUDA GPUs (-1 for default) */
	int ncuda;
	/* maximum number of Cell's SPUs (-1 for default) */
	int nspus;

	unsigned use_explicit_workers_bindid;
	unsigned workers_bindid[STARPU_NMAXWORKERS];

	unsigned use_explicit_workers_gpuid;
	unsigned workers_gpuid[STARPU_NMAXWORKERS];

	/* calibrate performance models, if any */
	unsigned calibrate;
};

/* Initialization method: it must be called prior to any other StarPU call
 * Default configuration is used if NULL is passed as argument.
 */
int starpu_init(struct starpu_conf *conf);

/* Shutdown method: note that statistics are only generated once StarPU is
 * shutdown */
void starpu_shutdown(void);

/* This function returns the number of workers (ie. processing units executing
 * StarPU tasks). The returned value should be at most STARPU_NMAXWORKERS. */
unsigned starpu_get_worker_count(void);

unsigned starpu_get_core_worker_count(void);
unsigned starpu_get_cuda_worker_count(void);
unsigned starpu_get_spu_worker_count(void);

/* Return the identifier of the thread in case this is associated to a worker.
 * This will return -1 if this function is called directly from the application
 * or if it is some SPU worker where a single thread controls different SPUs. */
int starpu_get_worker_id(void);

enum starpu_archtype {
	STARPU_CORE_WORKER, /* CPU core */
	STARPU_CUDA_WORKER, /* NVIDIA CUDA device */
	STARPU_GORDON_WORKER /* Cell SPU */
};

/* This function returns the type of worker associated to an identifier (as
 * returned by the starpu_get_worker_id function). The returned value indicates
 * the architecture of the worker: STARPU_CORE_WORKER for a CPU core,
 * STARPU_CUDA_WORKER for a CUDA device, and STARPU_GORDON_WORKER for a Cell
 * SPU. The value returned for an invalid identifier is unspecified.  */
enum starpu_archtype starpu_get_worker_type(int id);

/* StarPU associates a unique human readable string to each processing unit.
 * This function copies at most the "maxlen" first bytes of the unique
 * string associated to a worker identified by its identifier "id" into
 * the "dst" buffer. The caller is responsible for ensuring that the
 * "dst" is a valid pointer to a buffer of "maxlen" bytes at least.
 * Calling this function on an invalid identifier results in an unspecified
 * behaviour. */
void starpu_get_worker_name(int id, char *dst, size_t maxlen);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_H__
