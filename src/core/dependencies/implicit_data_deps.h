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

#ifndef __IMPLICIT_DATA_DEPS_H__
#define __IMPLICIT_DATA_DEPS_H__

/** @file */

#include <starpu.h>
#include <common/config.h>

#pragma GCC visibility push(hidden)

struct starpu_task *_starpu_detect_implicit_data_deps_with_handle(struct starpu_task *pre_sync_task, int *submit_pre_sync, struct starpu_task *post_sync_task, struct _starpu_task_wrapper_dlist *post_sync_task_dependency_slot,
								  starpu_data_handle_t handle, enum starpu_data_access_mode mode, unsigned task_handle_sequential_consistency);
int _starpu_test_implicit_data_deps_with_handle(starpu_data_handle_t handle, enum starpu_data_access_mode mode);
void _starpu_detect_implicit_data_deps(struct starpu_task *task);
void _starpu_release_data_enforce_sequential_consistency(struct starpu_task *task, struct _starpu_task_wrapper_dlist *task_dependency_slot, starpu_data_handle_t handle);
void _starpu_release_task_enforce_sequential_consistency(struct _starpu_job *j);

void _starpu_add_post_sync_tasks(struct starpu_task *post_sync_task, starpu_data_handle_t handle);
void _starpu_unlock_post_sync_tasks(starpu_data_handle_t handle, enum starpu_data_access_mode mode);

/** Register a hook to be called when a write is submitted */
void _starpu_implicit_data_deps_write_hook(void (*func)(starpu_data_handle_t)) STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

/** This function blocks until the handle is available in the requested mode */
int _starpu_data_wait_until_available(starpu_data_handle_t handle, enum starpu_data_access_mode mode, const char *sync_name);

void _starpu_data_clear_implicit(starpu_data_handle_t handle);

#pragma GCC visibility pop

#endif // __IMPLICIT_DATA_DEPS_H__

