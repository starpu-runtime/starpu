/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2025-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_CHECKPOINT_STORAGE_H__
#define __STARPU_MPI_CHECKPOINT_STORAGE_H__

#include <stddef.h>
#include <starpu_mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

	/* Initialise the storage backend pointing at the given shared directory.
	 * The directory is created if it does not exist.
	 * Returns 0 on success, -1 on error. */
	int _starpu_mpi_checkpoint_storage_init(const char *path);

	/* Tear down the storage backend. */
	int _starpu_mpi_checkpoint_storage_shutdown(void);

	/* Returns 1 if storage has been configured, 0 otherwise. */
	int _starpu_mpi_checkpoint_storage_is_enabled(void);

	/* Write a single data buffer for the given checkpoint tuple.
	 * Each item is identified by (cp_id, cp_inst, rank, tag).
	 * Returns 0 on success, -1 on error. */
	int _starpu_mpi_checkpoint_storage_write(int cp_id, int cp_inst, int rank, starpu_mpi_tag_t tag, const void *data, size_t size);

	/* Write the manifest file.  Must be called by rank 0 only, and only
	 * once every rank is known to have written its items successfully.
	 * Records cp_id, cp_inst, the number of MPI ranks, and the total number
	 * of data items persisted.  Returns 0 on success, -1 on error. */
	int _starpu_mpi_checkpoint_storage_write_manifest(int cp_id, int cp_inst, int n_ranks, int n_items);

	/* Atomically update the "latest" marker to (cp_id, cp_inst).
	 * Must be called by rank 0 only.  Returns 0 on success, -1 on error. */
	int _starpu_mpi_checkpoint_storage_update_latest(int cp_id, int cp_inst);

	/* Remove the checkpoint directory for (cp_id, cp_inst).
	 * Must be called by rank 0 only.  Safe to call even if directory is absent. */
	int _starpu_mpi_checkpoint_storage_delete(int cp_id, int cp_inst);

	/* Return the highest checkpoint instance of cp_id present in the
	 * storage directory, or -1 if there is none.  Used to pick the instance
	 * a new checkpoint is written to, so that the previous one is left
	 * untouched until the new one is committed. */
	int _starpu_mpi_checkpoint_storage_find_last_instance(int cp_id);

	/* Locate the latest validated checkpoint.
	 * Returns 0 and fills *cp_id / *cp_inst on success;
	 * returns -1 if no checkpoint exists yet. */
	int _starpu_mpi_checkpoint_storage_find_latest(int *cp_id, int *cp_inst);

	/* Read the number of MPI ranks recorded in the manifest for (cp_id, cp_inst).
	 * Returns the stored n_ranks on success, -1 on error. */
	int _starpu_mpi_checkpoint_storage_read_n_ranks(int cp_id, int cp_inst);

	/* Read a data item previously written by _starpu_mpi_checkpoint_storage_write.
	 * Allocates *data (caller must free) and sets *size_out.
	 * Returns 0 on success, -1 if the file is absent, negative errno on I/O error. */
	int _starpu_mpi_checkpoint_storage_read(int cp_id, int cp_inst, int rank, starpu_mpi_tag_t tag, void **data, size_t *size_out);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_MPI_CHECKPOINT_STORAGE_H__ */
