/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#include <datawizard/footprint.h>
#include <starpu_hash.h>
#include <core/task.h>
#include <starpu_scheduler.h>

uint32_t starpu_task_data_footprint(struct starpu_task *task)
{
	uint32_t footprint = 0;
	unsigned buffer;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, buffer);

		uint32_t handle_footprint = _starpu_data_get_footprint(handle);

		footprint = starpu_hash_crc32c_be(handle_footprint, footprint);
	}

	return footprint;
}

uint32_t _starpu_compute_buffers_footprint(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, unsigned nimpl, struct _starpu_job *j)
{
	if (j->footprint_is_computed)
		return j->footprint;

	uint32_t footprint = 0;

	struct starpu_task *task = j->task;

	if (model)
	{
		if (model->footprint)
		{
			footprint = model->footprint(task);
		}
		else
		{
			struct starpu_perfmodel_per_arch *per_arch;
			if (arch)
				per_arch = starpu_perfmodel_get_model_per_arch(model, arch, nimpl);
			if (arch && per_arch != NULL && per_arch->size_base)
			{
				size_t size = per_arch->size_base(task, arch, nimpl);
				footprint = starpu_hash_crc32c_be_n(&size, sizeof(size), footprint);
			}
			else if (model->size_base)
			{
				size_t size = model->size_base(task, nimpl);
				footprint = starpu_hash_crc32c_be_n(&size, sizeof(size), footprint);
			}
			else
			{
				footprint = starpu_task_data_footprint(task);
			}
		}
	}
	else
	{
		footprint = starpu_task_data_footprint(task);
	}

	j->footprint = footprint;
	j->footprint_is_computed = 1;

	return footprint;
}

uint32_t _starpu_compute_data_footprint(starpu_data_handle_t handle)
{
	uint32_t interfaceid = (uint32_t)starpu_data_get_interface_id(handle);

	STARPU_ASSERT(handle->ops->footprint);

	uint32_t handle_footprint = handle->ops->footprint(handle);

	return starpu_hash_crc32c_be(handle_footprint, interfaceid);
}

uint32_t _starpu_compute_data_alloc_footprint(starpu_data_handle_t handle)
{
	uint32_t interfaceid = (uint32_t)starpu_data_get_interface_id(handle);

	uint32_t handle_footprint;
	if (handle->ops->alloc_footprint)
		handle_footprint = handle->ops->alloc_footprint(handle);
	else
		handle_footprint = handle->ops->footprint(handle);

	return starpu_hash_crc32c_be(handle_footprint, interfaceid);
}

uint32_t starpu_task_footprint(struct starpu_perfmodel *model, struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	return _starpu_compute_buffers_footprint(model, arch, nimpl, j);
}
