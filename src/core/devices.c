/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <core/devices.h>

struct _starpu_device_entry
{
	UT_hash_handle hh;
	unsigned deviceid;
};

static struct _starpu_device_entry *gpu_devices_already_used;

void _starpu_devices_gpu_set_used(int devid)
{
	struct _starpu_device_entry *entry;
	HASH_FIND_INT(gpu_devices_already_used, &devid, entry);
	if (!entry)
	{
		_STARPU_MALLOC(entry, sizeof(*entry));
		entry->deviceid = devid;
		HASH_ADD_INT(gpu_devices_already_used, deviceid, entry);
	}
}

void _starpu_devices_gpu_clear(struct _starpu_machine_config *config, enum starpu_worker_archtype type)
{
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned tmp[STARPU_NMAXWORKERS];
	unsigned nb=0;
	int i;
	for(i=0 ; i<STARPU_NMAXWORKERS ; i++)
	{
		struct _starpu_device_entry *entry;
		int devid = topology->workers_devid[type][i];

		HASH_FIND_INT(gpu_devices_already_used, &devid, entry);
		if (entry == NULL)
		{
			tmp[nb] = devid;
			nb++;
		}
	}
	for (i=nb ; i<STARPU_NMAXWORKERS ; i++)
		tmp[i] = -1;
	memcpy(topology->workers_devid[type], tmp, sizeof(unsigned)*STARPU_NMAXWORKERS);
}

void _starpu_devices_drop_duplicate(unsigned ids[STARPU_NMAXWORKERS])
{
	struct _starpu_device_entry *devices_already_used = NULL;
	unsigned tmp[STARPU_NMAXWORKERS];
	unsigned nb=0;
	int i;

	for(i=0 ; i<STARPU_NMAXWORKERS ; i++)
	{
		int devid = ids[i];
		struct _starpu_device_entry *entry;
		HASH_FIND_INT(devices_already_used, &devid, entry);
		if (entry == NULL)
		{
			struct _starpu_device_entry *entry2;
			_STARPU_MALLOC(entry2, sizeof(*entry2));
			entry2->deviceid = devid;
			HASH_ADD_INT(devices_already_used, deviceid,  entry2);
			tmp[nb] = devid;
			nb ++;
		}
	}
	struct _starpu_device_entry *entry=NULL, *tempo=NULL;
	HASH_ITER(hh, devices_already_used, entry, tempo)
	{
		HASH_DEL(devices_already_used, entry);
		free(entry);
	}
	for (i=nb ; i<STARPU_NMAXWORKERS ; i++)
		tmp[i] = -1;
	memcpy(ids, tmp, sizeof(unsigned)*STARPU_NMAXWORKERS);
}

void _starpu_devices_gpu_clean()
{
	struct _starpu_device_entry *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, gpu_devices_already_used, entry, tmp)
	{
		HASH_DEL(gpu_devices_already_used, entry);
		free(entry);
	}
	gpu_devices_already_used = NULL;
}
