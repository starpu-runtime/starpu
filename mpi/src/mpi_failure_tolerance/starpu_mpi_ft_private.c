/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi_private.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_template.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_package.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_service_comms.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_stats.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_storage.h>

starpu_pthread_mutex_t           ft_mutex;
int                              _my_rank;
static int                       _restart_cp_id   = -1;
static int                       _restart_cp_inst = -1;
static int                       _restart_n_ranks = -1;

int starpu_mpi_checkpoint_init(void)
{
	STARPU_PTHREAD_MUTEX_INIT(&ft_mutex, NULL);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &_my_rank); //TODO: check compatibility with several Comms behaviour
	starpu_mpi_ft_service_lib_init(_ack_msg_recv_cb, _cp_discard_message_recv_cb);
	checkpoint_template_lib_init();
	_starpu_mpi_checkpoint_tracker_init();
	checkpoint_package_init();
	_STARPU_MPI_FT_STATS_INIT();
	return 0;
}

int starpu_mpi_checkpoint_shutdown(void)
{
	checkpoint_template_lib_quit();
	checkpoint_package_shutdown();
	_starpu_mpi_checkpoint_tracker_shutdown();
	_starpu_mpi_checkpoint_storage_shutdown();
	STARPU_PTHREAD_MUTEX_DESTROY(&ft_mutex);
	_STARPU_MPI_FT_STATS_WRITE_TO_FD(stderr);
	_STARPU_MPI_FT_STATS_SHUTDOWN();
	return 0;
}

int starpu_mpi_checkpoint_set_storage_path(const char *path)
{
	return _starpu_mpi_checkpoint_storage_init(path);
}

int starpu_mpi_checkpoint_flush_to_storage(starpu_mpi_checkpoint_template_t cp_template)
{
	if (!_starpu_mpi_checkpoint_storage_is_enabled())
	{
		_STARPU_DISP("[storage] flush called but storage path not configured\n");
		return -1;
	}

	int my_rank, comm_size_val;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size_val);

	/* Deliberately NOT starpu_task_wait_for_all() here. Coherence is already
	 * guaranteed per entry: STARPU_VALUE entries are written from item->ptr,
	 * which the runtime never touches, and STARPU_R entries are read below
	 * under starpu_data_acquire(), which waits until the handle is coherent in
	 * main RAM. Waiting on the whole task graph would moreover hold the
	 * checkpoint back until the tasks in flight finish, which defeats a
	 * checkpoint that has to be taken quickly. Callers who want a quiescent
	 * call starpu_task_wait_for_all() themselves before flushing.
	 */

	int cp_id = cp_template->cp_id;

	/* Every node must write into the same checkpoint instance, and that
	 * instance must not be one already committed: the previous checkpoint
	 * has to stay intact until this one is complete. The node of rank 0
	 * picks the instance from what the storage already holds, and tells the
	 * others. */
	int cp_inst = 0;
	if (my_rank == 0)
		cp_inst = _starpu_mpi_checkpoint_storage_find_last_instance(cp_id) + 1;
	MPI_Bcast(&cp_inst, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int n_items = 0;
	int n_errors = 0;
	int n_skipped = 0;

	struct _starpu_mpi_checkpoint_template_item *item =
	        _starpu_mpi_checkpoint_template_get_first_data(cp_template);

	while (item != _starpu_mpi_checkpoint_template_end(cp_template))
	{
		/* Entries this node holds a backup of belong to another node, which
		 * writes them itself. */
		if (item->backup_of == -1 && item->backupped_by == -1)
		{
			/* Owned, but no node was given to back it up. Such an entry
			 * is not part of the checkpoint, and an application which
			 * registered its data without backups would otherwise get an
			 * empty checkpoint without being told. */
			n_skipped++;
		}
		else if (item->backup_of == -1)
		{
			if (item->type == STARPU_VALUE)
			{
				int ret = _starpu_mpi_checkpoint_storage_write(
				        cp_id, cp_inst, my_rank, item->tag,
				        item->ptr, (size_t)item->count);
				if (ret == 0)
					n_items++;
				else
					n_errors++;
			}
			else if (item->type == STARPU_R)
			{
				starpu_data_handle_t handle = (starpu_data_handle_t)item->ptr;
				starpu_mpi_tag_t tag = starpu_mpi_data_get_tag(handle);

				/* Acquire: guarantees data is coherent in main RAM */
				/* TODO: we could try to use starpu_data_acquire_cb to write
				 * data as it gets computed rather than strictly in the template
				 * order */
				int ret = starpu_data_acquire(handle, STARPU_R);
				if (ret != 0)
				{
					_STARPU_DISP("[storage] Cannot acquire the handle of tag %lld: %d\n",
					             (long long)tag, ret);
					n_errors++;
				}
				else
				{
					void *ptr = starpu_data_handle_to_pointer(handle, STARPU_MAIN_RAM);
					size_t sz = starpu_data_get_size(handle);
					ret = _starpu_mpi_checkpoint_storage_write(
					        cp_id, cp_inst, my_rank, tag, ptr, sz);
					starpu_data_release(handle);
					if (ret == 0)
						n_items++;
					else
						n_errors++;
				}
			}
		}
		item = _starpu_mpi_checkpoint_template_get_next_data(cp_template, item);
	}

	/* Collect what every node achieved. This also acts as the barrier which
	 * makes sure they are all done writing, but unlike a plain barrier it
	 * tells rank 0 whether the checkpoint is worth committing: a checkpoint
	 * missing entries must never be published as the latest valid one, since
	 * restarting from it would silently lose data. */
	if (n_skipped > 0)
		_STARPU_DISP("[storage] %d entr%s of checkpoint %d left out: no backup node was given for them\n",
		             n_skipped, n_skipped > 1 ? "ies" : "y", cp_id);

	int total_errors = 0;
	int total_items = 0;
	MPI_Allreduce(&n_errors, &total_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&n_items, &total_items, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	if (total_errors > 0)
	{
		/* Leave the previous checkpoint and its marker alone, and remove
		 * what was written of this one. */
		if (my_rank == 0)
			_starpu_mpi_checkpoint_storage_delete(cp_id, cp_inst);
		MPI_Barrier(MPI_COMM_WORLD);

		_STARPU_DISP("[storage] Checkpoint cp_id=%d cp_inst=%d discarded: %d item(s) could not be written\n",
		             cp_id, cp_inst, total_errors);
		return -1;
	}

	int commit_failed = 0;
	if (my_rank == 0)
	{
		if (_starpu_mpi_checkpoint_storage_write_manifest(cp_id, cp_inst, comm_size_val, total_items) != 0
		    || _starpu_mpi_checkpoint_storage_update_latest(cp_id, cp_inst) != 0)
		{
			_starpu_mpi_checkpoint_storage_delete(cp_id, cp_inst);
			commit_failed = 1;
		}
		else if (cp_inst >= 1)
		{
			/* The new checkpoint is committed, the previous one is no
			 * longer needed. */
			_starpu_mpi_checkpoint_storage_delete(cp_id, cp_inst - 1);
		}
	}

	/* No node may go on to restart, and read a checkpoint directory which is
	 * not committed yet, before rank 0 is done. */
	MPI_Bcast(&commit_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (commit_failed)
	{
		_STARPU_DISP("[storage] Checkpoint cp_id=%d cp_inst=%d discarded: could not commit\n",
		             cp_id, cp_inst);
		return -1;
	}

	_STARPU_MPI_DEBUG(0, "[storage] Checkpoint flushed: cp_id=%d cp_inst=%d n_items=%d\n", cp_id, cp_inst, total_items);
	return 0;
}

int starpu_mpi_init_from_checkpoint(const char *storage_path, int *restart_flag)
{
	*restart_flag = 0;
	_restart_cp_id = -1;
	_restart_cp_inst = -1;
	_restart_n_ranks = -1;

	if (!storage_path || storage_path[0] == '\0') return 0;

	if (_starpu_mpi_checkpoint_storage_init(storage_path) != 0) return -1;

	int cp_id = -1, cp_inst = -1;
	if (_starpu_mpi_checkpoint_storage_find_latest(&cp_id, &cp_inst) != 0)
	{
		_STARPU_MPI_DEBUG(0, "[storage] No previous checkpoint found in '%s'\n", storage_path);
		return 0;
	}

	int n_ranks = _starpu_mpi_checkpoint_storage_read_n_ranks(cp_id, cp_inst);

	_restart_cp_id = cp_id;
	_restart_cp_inst = cp_inst;
	_restart_n_ranks = n_ranks;
	*restart_flag = 1;

	_STARPU_MPI_DEBUG(0, "[storage] Restart checkpoint found: cp_id=%d cp_inst=%d n_ranks=%d\n", cp_id, cp_inst, n_ranks);
	return 0;
}

int starpu_mpi_checkpoint_get_restart_cp_id(void)
{
	return _restart_cp_id;
}

int starpu_mpi_checkpoint_get_restart_cp_inst(void)
{
	return _restart_cp_inst;
}

int starpu_mpi_checkpoint_get_restart_n_ranks(void)
{
	return _restart_n_ranks;
}

int starpu_mpi_checkpoint_restore_handle(int cp_id, int cp_inst, int old_rank, starpu_mpi_tag_t tag, starpu_data_handle_t handle)
{
	void *data = NULL;
	size_t sz = 0;
	int ret = _starpu_mpi_checkpoint_storage_read(cp_id, cp_inst, old_rank, tag, &data, &sz);
	if (ret != 0) return ret;

	/* Copy data into the handle's main-RAM buffer */
	ret = starpu_data_acquire(handle, STARPU_W);
	if (ret != 0)
	{
		_STARPU_DISP("[storage] restore_handle: cannot acquire the handle: %d\n", ret);
		free(data);
		return -1;
	}

	void *dst = starpu_data_handle_to_pointer(handle, STARPU_MAIN_RAM);
	size_t handle_sz = starpu_data_get_size(handle);

	if (dst && sz <= handle_sz)
	{
		memcpy(dst, data, sz);
	}
	else
	{
		_STARPU_DISP("[storage] restore_handle: size mismatch or null pointer (handle_sz=%zu, file_sz=%zu)\n", handle_sz, sz);
		starpu_data_release(handle);
		free(data);
		return -1;
	}

	starpu_data_release(handle);
	free(data);
	return 0;
}

int starpu_mpi_checkpoint_restore_value(int cp_id, int cp_inst, int old_rank, starpu_mpi_tag_t tag, void *ptr, size_t size)
{
	void *data = NULL;
	size_t sz = 0;
	int ret = _starpu_mpi_checkpoint_storage_read(cp_id, cp_inst, old_rank, tag, &data, &sz);
	if (ret != 0) return ret;

	if (sz != size)
	{
		_STARPU_DISP("[storage] restore_value: size mismatch (expected %zu, got %zu)\n", size, sz);
		free(data);
		return -1;
	}

	memcpy(ptr, data, sz);
	free(data);
	return 0;
}

void starpu_mpi_ft_progress(void)
{
	starpu_mpi_ft_service_progress();
}

int starpu_mpi_ft_busy()
{
	return starpu_mpi_ft_service_lib_busy();
}
