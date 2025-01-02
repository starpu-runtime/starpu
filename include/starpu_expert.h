/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_EXPERT_H__
#define __STARPU_EXPERT_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Expert_Mode Expert Mode
   @{
*/

/**
   Wake all the workers, so they can inspect data requests and task
   submissions again.
*/
void starpu_wake_all_blocked_workers(void);

/**
   Register a progression hook, to be called when workers are idle.
*/
int starpu_progression_hook_register(unsigned (*func)(void *arg), void *arg);

/**
   Unregister a given progression hook.
*/
void starpu_progression_hook_deregister(int hook_id);

int starpu_idle_hook_register(unsigned (*func)(void *arg), void *arg);
void starpu_idle_hook_deregister(int hook_id);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_H__ */
