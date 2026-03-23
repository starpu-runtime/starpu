/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SCHEDULERS_HPLOOKAHEAD_H__
#define __STARPU_SCHEDULERS_HPLOOKAHEAD_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* Presently Assume that there won't be more than 100 types of
 * tasks in any application (may not be a reasonable assumption).
 *
 * TODO: remove dependency on number of number of different types of tasks
 *
 */
#define STARPU_HPLOOKAHEAD_MAXTYPESOFTASKS 10
#define STARPU_HPLOOKAHEAD_NTASKSPERQUEUEINSIMULATION 400

//int push_task_in_to_ready_queue(struct starpu_task *task);
//struct starpu_task* pop_task_from_ready_queue(unsigned sched_ctx_id);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHEDULERS_HPLOOKAHEAD_H__ */

