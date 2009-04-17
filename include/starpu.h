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

#include <starpu_config.h>
#include <starpu-util.h>
#include <starpu-data.h>
#include <starpu-perfmodel.h>
#include <starpu-task.h>

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

	/* calibrate performance models, if any */
	unsigned calibrate;
};

/* Initialization method: it must be called prior to any other StarPU call
 * Default configuration is used if NULL is passed as argument.
 */
void starpu_init(struct starpu_conf *conf);

/* Shutdown method: note that statistics are only generated once StarPU is
 * shutdown */
void starpu_shutdown(void);

#endif // __STARPU_H__
