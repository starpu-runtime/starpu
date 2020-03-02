/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2012       Vincent Danjean
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

#include "socl.h"
#include "init.h"

/**
 * \brief Return one device of each kind
 *
 * \param[in] platform Must be StarPU platform ID or NULL
 */
CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclGetDeviceIDs(cl_platform_id   platform,
		 cl_device_type   device_type,
		 cl_uint          num_entries,
		 cl_device_id *   devices,
		 cl_uint *        num_devices)
{
	if (socl_init_starpu() < 0)
	{
		*num_devices = 0;
		return CL_SUCCESS;
	}

	if (_starpu_init_failed)
	{
		*num_devices = 0;
		return CL_SUCCESS;
	}

	if (platform != NULL && platform != &socl_platform)
		return CL_INVALID_PLATFORM;

	if ((devices != NULL && num_entries == 0)
	    || (devices == NULL && num_devices == NULL))
		return CL_INVALID_VALUE;

	if (!(device_type & (CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_DEFAULT))
	    && (device_type != CL_DEVICE_TYPE_ALL))
		return CL_INVALID_DEVICE_TYPE;

	int ndevs = starpu_worker_get_count_by_type(STARPU_OPENCL_WORKER);

	int workers[ndevs];
	starpu_worker_get_ids_by_type(STARPU_OPENCL_WORKER, workers, ndevs);

	if (socl_devices == NULL)
	{
		socl_device_count = ndevs;
		socl_devices = malloc(sizeof(struct _cl_device_id) * ndevs);
		int i;
		for (i=0; i < ndevs; i++)
		{
			int devid = starpu_worker_get_devid(workers[i]);
			socl_devices[i].dispatch = &socl_master_dispatch;
			socl_devices[i].worker_id = workers[i];
			socl_devices[i].device_id = devid;
		}
	}

	int i;
	unsigned int num = 0;
	for (i=0; i < ndevs; i++)
	{
		int devid = socl_devices[i].device_id;
		cl_device_id dev;
		starpu_opencl_get_device(devid, &dev);
		cl_device_type typ;
		clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(typ), &typ, NULL);
		if (typ & device_type)
		{
			if (devices != NULL && num < num_entries) devices[num] = &socl_devices[i];
			num++;
		}
	}

	if (num_devices != NULL)
		*num_devices = num;

	return CL_SUCCESS;
}
