/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DATA_CONCURRENCY_H__
#define __DATA_CONCURRENCY_H__

/** @file */

#include <core/jobs.h>

void _starpu_job_set_ordered_buffers(struct _starpu_job *j);

unsigned _starpu_submit_job_enforce_data_deps(struct _starpu_job *j);
void _starpu_submit_job_enforce_arbitered_deps(struct _starpu_job *j, unsigned buf, unsigned nbuffers);
void _starpu_enforce_data_deps_notify_job_ready_soon(struct _starpu_job *j, _starpu_notify_job_start_data *data);

int _starpu_notify_data_dependencies(starpu_data_handle_t handle);
void _starpu_notify_arbitered_dependencies(starpu_data_handle_t handle);

unsigned _starpu_attempt_to_submit_data_request_from_apps(starpu_data_handle_t handle,
							  enum starpu_data_access_mode mode,
							  void (*callback)(void *), void *argcb);

unsigned _starpu_attempt_to_submit_arbitered_data_request(unsigned request_from_codelet,
						       starpu_data_handle_t handle, enum starpu_data_access_mode mode,
						       void (*callback)(void *), void *argcb,
						       struct _starpu_job *j, unsigned buffer_index);

#endif // __DATA_CONCURRENCY_H__

