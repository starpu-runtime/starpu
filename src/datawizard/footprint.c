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

#include <datawizard/footprint.h>
#include <common/hash.h>

void _starpu_compute_buffers_footprint(starpu_job_t j)
{
	uint32_t footprint = 0;
	unsigned buffer;

	struct starpu_task *task = j->task;

	for (buffer = 0; buffer < task->cl->nbuffers; buffer++)
	{
		starpu_data_handle handle = task->buffers[buffer].handle;

		STARPU_ASSERT(handle->ops);
		STARPU_ASSERT(handle->ops->footprint);

		uint32_t handle_footprint = handle->ops->footprint(handle);

		footprint = _starpu_crc32_be(handle_footprint, footprint);
	}

	j->footprint = footprint;
	j->footprint_is_computed = 1;
}

inline uint32_t _starpu_compute_data_footprint(starpu_data_handle handle)
{
	uint32_t interfaceid = (uint32_t)starpu_get_handle_interface_id(handle);

	uint32_t handle_footprint = handle->ops->footprint(handle);

	return _starpu_crc32_be(handle_footprint, interfaceid);
}
