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

#ifndef __IMPLICIT_DATA_DEPS_H__
#define __IMPLICIT_DATA_DEPS_H__

#include <starpu.h>
#include <common/config.h>

void _starpu_detect_implicit_data_deps_with_handle(struct starpu_task *pre_sync_task, struct starpu_task *post_sync_task,
						starpu_data_handle handle, starpu_access_mode mode);
void _starpu_detect_implicit_data_deps(struct starpu_task *task);
void _starpu_release_data_enforce_sequential_consistency(struct starpu_task *task, starpu_data_handle handle);

void _starpu_add_post_sync_tasks(struct starpu_task *post_sync_task, starpu_data_handle handle);
void _starpu_unlock_post_sync_tasks(starpu_data_handle handle);

#endif // __IMPLICIT_DATA_DEPS_H__

