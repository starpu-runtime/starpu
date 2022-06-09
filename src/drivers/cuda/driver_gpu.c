
#include <drivers/cuda/driver_gpu.h>

/* Entry in the `devices_using_cuda' hash table.  */
static struct _starpu_gpu_entry *devices_already_used;

void _starpu_gpu_set_used(int devid)
{
	struct _starpu_gpu_entry *entry;
	HASH_FIND_INT(devices_already_used, &devid, entry);
	if (!entry)
	{
		_STARPU_MALLOC(entry, sizeof(*entry));
		entry->gpuid = devid;
		HASH_ADD_INT(devices_already_used, gpuid, entry);
	}
}

void _starpu_gpu_clear(struct _starpu_machine_config *config, enum starpu_worker_archtype type)
{
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned tmp[STARPU_NMAXWORKERS];
	unsigned nb=0;
	int i;
	for(i=0 ; i<STARPU_NMAXWORKERS ; i++)
	{
		struct _starpu_gpu_entry *entry;
		int devid = config->topology.workers_devid[type][i];

		HASH_FIND_INT(devices_already_used, &devid, entry);
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

void _starpu_gpu_clean()
{
	struct _starpu_gpu_entry *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, devices_already_used, entry, tmp)
	{
		HASH_DEL(devices_already_used, entry);
		free(entry);
	}
	devices_already_used = NULL;
}
