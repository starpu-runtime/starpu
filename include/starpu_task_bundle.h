/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011,2014                           Université de Bordeaux
 * Copyright (C) 2011,2012                                Inria
 * Copyright (C) 2011-2013,2015,2017                      CNRS
 * Copyright (C) 2011                                     Télécom-SudParis
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

#ifndef __STARPU_TASK_BUNDLE_H__
#define __STARPU_TASK_BUNDLE_H__

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_task;

typedef struct _starpu_task_bundle *starpu_task_bundle_t;

void starpu_task_bundle_create(starpu_task_bundle_t *bundle);

int starpu_task_bundle_insert(starpu_task_bundle_t bundle, struct starpu_task *task);

int starpu_task_bundle_remove(starpu_task_bundle_t bundle, struct starpu_task *task);

void starpu_task_bundle_close(starpu_task_bundle_t bundle);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_BUNDLE_H__ */
