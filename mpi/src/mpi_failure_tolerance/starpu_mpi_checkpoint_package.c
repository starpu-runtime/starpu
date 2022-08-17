/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <mpi_failure_tolerance/starpu_mpi_checkpoint_package.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_stats.h>

struct _starpu_mpi_checkpoint_data_list* checkpoint_data_list;
starpu_pthread_mutex_t package_package_mutex;

int _checkpoint_package_data_delete_all();

int checkpoint_package_init()
{
	STARPU_PTHREAD_MUTEX_INIT(&package_package_mutex, NULL);
	checkpoint_data_list = _starpu_mpi_checkpoint_data_list_new();
	_starpu_mpi_checkpoint_data_list_init(checkpoint_data_list);
	return 0;
}

int checkpoint_package_shutdown()
{
	_checkpoint_package_data_delete_all();
	STARPU_PTHREAD_MUTEX_DESTROY(&package_package_mutex);
	return 0;
}

#ifdef STARPU_USE_MPI_FT_STATS
void _stats_store_checkpoint_data(struct _starpu_mpi_checkpoint_data* new_checkpoint_data)
{
	struct _starpu_mpi_checkpoint_data* next_checkpoint_data;
	struct _starpu_mpi_checkpoint_data* checkpoint_data = _starpu_mpi_checkpoint_data_list_begin(checkpoint_data_list);
	while (checkpoint_data != _starpu_mpi_checkpoint_data_list_end(checkpoint_data_list))
	{
		next_checkpoint_data = _starpu_mpi_checkpoint_data_list_next(checkpoint_data);
		if (checkpoint_data->tag == new_checkpoint_data->tag && checkpoint_data->ptr == new_checkpoint_data->ptr)
		{
			// The data is already in the CP data list,don't count it as a new data
			return;
		}
		checkpoint_data = next_checkpoint_data;
	}
	_STARPU_MPI_FT_STATS_STORE_CP_DATA(new_checkpoint_data->type==STARPU_VALUE?new_checkpoint_data->count:new_checkpoint_data->type==STARPU_R?starpu_data_get_size((starpu_data_handle_t) new_checkpoint_data->ptr):-1);
}
#else
void _stats_store_checkpoint_data(STARPU_ATTRIBUTE_UNUSED struct _starpu_mpi_checkpoint_data* new_checkpoint_data)
{
	return;
}
#endif

#ifdef STARPU_USE_MPI_FT_STATS
void _stats_discard_checkpoint_data(struct _starpu_mpi_checkpoint_data* new_checkpoint_data)
{
	struct _starpu_mpi_checkpoint_data* next_checkpoint_data;
	struct _starpu_mpi_checkpoint_data* checkpoint_data = _starpu_mpi_checkpoint_data_list_begin(checkpoint_data_list);
	while (checkpoint_data != _starpu_mpi_checkpoint_data_list_end(checkpoint_data_list))
	{
		next_checkpoint_data = _starpu_mpi_checkpoint_data_list_next(checkpoint_data);
		if (checkpoint_data->tag == new_checkpoint_data->tag && checkpoint_data->ptr == new_checkpoint_data->ptr)
		{
			// The data is still in the CP data list, don't count it as a discard
			return;
		}
		checkpoint_data = next_checkpoint_data;
	}
	_STARPU_MPI_FT_STATS_DISCARD_CP_DATA(new_checkpoint_data->type==STARPU_VALUE?new_checkpoint_data->count:new_checkpoint_data->type==STARPU_R?starpu_data_get_size((starpu_data_handle_t) new_checkpoint_data->ptr):-1);
}
#else
void _stats_discard_checkpoint_data(STARPU_ATTRIBUTE_UNUSED struct _starpu_mpi_checkpoint_data* new_checkpoint_data)
{
	return;
}
#endif

int checkpoint_package_data_add(int cp_id, int cp_inst, int rank, starpu_mpi_tag_t tag, int type, void* ptr, int count)
{
	struct _starpu_mpi_checkpoint_data* checkpoint_data = _starpu_mpi_checkpoint_data_new();
	checkpoint_data->cp_id = cp_id;
	checkpoint_data->cp_inst = cp_inst;
	checkpoint_data->rank = rank;
	checkpoint_data->tag = tag;
	checkpoint_data->type = type;
	checkpoint_data->ptr = ptr;
	checkpoint_data->count = count;
	STARPU_PTHREAD_MUTEX_LOCK(&package_package_mutex);
	_stats_store_checkpoint_data(checkpoint_data);
	_starpu_mpi_checkpoint_data_list_push_back(checkpoint_data_list, checkpoint_data);
	STARPU_PTHREAD_MUTEX_UNLOCK(&package_package_mutex);
	_STARPU_MPI_DEBUG(8, "CP data (%p) added - cpid:%d - cpinst:%d - rank:%d - tag:%ld\n", checkpoint_data->ptr, checkpoint_data->cp_id, checkpoint_data->cp_inst, checkpoint_data->rank, checkpoint_data->tag);
	return 0;
}

int _checkpoint_package_data_delete(struct _starpu_mpi_checkpoint_data* checkpoint_data)
{
	size_t size;
	_starpu_mpi_checkpoint_data_list_erase(checkpoint_data_list, checkpoint_data);
	_stats_discard_checkpoint_data(checkpoint_data);
	if (checkpoint_data->type==STARPU_R)
	{
		starpu_data_handle_t handle = checkpoint_data->ptr;
		size = starpu_data_get_size(handle);
		_STARPU_MPI_DEBUG(8, "Clearing handle %p entry\n", handle);
		starpu_data_unregister_submit(handle);
	}
	else if (checkpoint_data->type==STARPU_VALUE)
	{
		size = checkpoint_data->count;
		_STARPU_MPI_DEBUG(8, "Clearing external data entry\n");
		free(checkpoint_data->ptr);
	}
	else
	{
		STARPU_ABORT_MSG("Unrecognized data type: %d\n", checkpoint_data->type);
	}
	free(checkpoint_data);
	return size;
}

int checkpoint_package_data_del(int cp_id, int cp_inst, int rank)
{
	(void)cp_id;
	int done = 0;
	size_t size = 0;
	struct _starpu_mpi_checkpoint_data* next_checkpoint_data = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&package_package_mutex);
	struct _starpu_mpi_checkpoint_data* checkpoint_data = _starpu_mpi_checkpoint_data_list_begin(checkpoint_data_list);
	while (checkpoint_data != _starpu_mpi_checkpoint_data_list_end(checkpoint_data_list))
	{
		next_checkpoint_data = _starpu_mpi_checkpoint_data_list_next(checkpoint_data);
		// I delete all the old data (i.e. the cp inst is strictly lower than the one of the just validated CP) only for
		// the rank that initiated the CP
		if (checkpoint_data->cp_inst<cp_inst && checkpoint_data->rank==rank)
		{
			size += _checkpoint_package_data_delete(checkpoint_data);
			done++;
		}
		checkpoint_data = next_checkpoint_data;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&package_package_mutex);
	_STARPU_MPI_DEBUG(0, "cleared %d data from checkpoint database (%ld bytes).\n", done, size);

	return done;
}

int _checkpoint_package_data_delete_all()
{
	int done = 0;
	size_t size = 0;
	struct _starpu_mpi_checkpoint_data* next_checkpoint_data = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&package_package_mutex);
	struct _starpu_mpi_checkpoint_data* checkpoint_data = _starpu_mpi_checkpoint_data_list_begin(checkpoint_data_list);
	while (checkpoint_data != _starpu_mpi_checkpoint_data_list_end(checkpoint_data_list))
	{
		next_checkpoint_data = _starpu_mpi_checkpoint_data_list_next(checkpoint_data);
		// I delete all the data
		size += _checkpoint_package_data_delete(checkpoint_data);
		done++;
		checkpoint_data = next_checkpoint_data;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&package_package_mutex);
	_STARPU_MPI_DEBUG(0, "cleared %d data from checkpoint database (%ld bytes).\n", done, size);

	return done;
}
