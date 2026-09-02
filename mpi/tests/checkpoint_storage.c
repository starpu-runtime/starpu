/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <starpu_mpi.h>
#include <starpu_mpi_ft.h>
#include "helper.h"

/*
 * Test for persistent checkpoint storage (starpu_mpi_checkpoint_storage.c).
 *
 * The test runs in two phases inside a single process, simulating a job
 * that checkpoints, terminates, and then restarts:
 *
 *   Phase 1 – WRITE
 *     Each rank registers a float-vector handle and an int counter.
 *     The vector is owned by rank 0; the counter is per-rank.
 *     starpu_mpi_checkpoint_flush_to_storage() persists both to the
 *     storage directory.
 *
 *   Phase 2 – RESTORE
 *     starpu_mpi_init_from_checkpoint() reads the 'latest' file.
 *     restore_handle() reloads the vector on rank 0.
 *     restore_value() reloads each rank's counter, and then the counters
 *     written by the other ranks, to cover the case of a node taking over
 *     the entries of a node which is gone.
 *     All are checked against the values written in Phase 1.
 *
 * Needs at least 2 MPI ranks, and a storage directory shared by all of them:
 * STARPU_CP_STORAGE_PATH overrides the default below, which only suits a run
 * confined to one node.
 */


#define CP_STORAGE_PATH_DEFAULT "/tmp/starpu_cp_storage_test"
#define CP_ID 42
#define VEL_TAG ((starpu_mpi_tag_t)10)
#define COUNTER_TAG ((starpu_mpi_tag_t)1000)
#define ARRAY_SIZE 8

static int g_size = 2;
static const char *g_storage_path = CP_STORAGE_PATH_DEFAULT;

/* The storage backend expects a directory shared by all the nodes. When the
 * test is run on a single node the default in /tmp is enough, but a run
 * spanning several nodes needs a directory on a shared filesystem, given
 * through STARPU_CP_STORAGE_PATH. */
static const char * storage_path(void)
{
	const char *path = getenv("STARPU_CP_STORAGE_PATH");
	return (path && path[0] != '\0') ? path : CP_STORAGE_PATH_DEFAULT;
}

static int counter_backup_rank(int rank)
{
	return (rank + 1) % g_size;
}

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */

static void starpu_init_noworker(struct starpu_conf *conf)
{
	starpu_conf_init(conf);
	starpu_conf_noworker(conf);
	conf->ncpus = -1;
	conf->nmpi_sc = -1;
	conf->ntcpip_sc = -1;
}

/* Remove the checkpoint directory tree after the test. All the ranks must
 * call this, and they all wait before anything is removed: rank 0 must not
 * delete the checkpoint while another rank is still reading it. */
static void cleanup_storage(int rank)
{
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0)
	{
		DIR *top = opendir(g_storage_path);
		if (top)
		{
			const struct dirent *entry;
			while ((entry = readdir(top)) != NULL)
			{
				char path[1024];
				if (entry->d_name[0] == '.')
					continue;
				snprintf(path, sizeof(path), "%s/%s", g_storage_path, entry->d_name);
				if (unlink(path) == 0)
					continue;
				/* a checkpoint instance directory */
				DIR *sub = opendir(path);
				if (sub)
				{
					const struct dirent *sub_entry;
					while ((sub_entry = readdir(sub)) != NULL)
					{
						char sub_path[2048];
						if (sub_entry->d_name[0] == '.')
							continue;
						snprintf(sub_path, sizeof(sub_path), "%s/%s", path, sub_entry->d_name);
						unlink(sub_path);
					}
					closedir(sub);
					rmdir(path);
				}
			}
			closedir(top);
			rmdir(g_storage_path);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

/* -----------------------------------------------------------------------
 * Phase 1: write a checkpoint
 * ----------------------------------------------------------------------- */

static int phase_write(int argc, char *argv[], int rank, int size)
{
	struct starpu_conf conf;
	starpu_init_noworker(&conf);

	int ret = starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "[write] no device, skipping\n");
		return 77; /* skip code */
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_checkpoint_init();

	/* ---- data ---- */
	/* Velocity vector: owned by rank 0, value = rank*10 + index */
	float vel[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
		vel[i] = (float)(0 * 10 + i); /* rank-0 values */

	/* Each rank has a per-rank counter */
	int counter = 100 + rank;

	starpu_data_handle_t vel_handle;
	starpu_vector_data_register(&vel_handle, STARPU_MAIN_RAM,
	                            (uintptr_t)vel, ARRAY_SIZE, sizeof(float));
	starpu_mpi_data_register(vel_handle, VEL_TAG, 0); /* owner rank 0 */

	/* ---- checkpoint template ---- */
	starpu_mpi_checkpoint_template_t cp;
	int vel_backup = (size > 1) ? 1 : 0;

	starpu_mpi_checkpoint_template_register(&cp, CP_ID, 0,
	                                        STARPU_R, vel_handle, vel_backup,
	                                        STARPU_VALUE, &counter, sizeof(int),
	                                        COUNTER_TAG, counter_backup_rank,
	                                        0);

	/* ---- flush ---- */
	starpu_mpi_checkpoint_set_storage_path(g_storage_path);
	ret = starpu_mpi_checkpoint_flush_to_storage(cp);
	if (ret != 0)
	{
		FPRINTF(stderr, "[write][rank %d] ERROR: flush_to_storage returned %d\n",
		        rank, ret);
	}
	else
	{
		FPRINTF(stderr, "[write][rank %d] flush OK  (counter=%d)\n",
		        rank, counter);
	}

	starpu_data_unregister(vel_handle);
	starpu_mpi_checkpoint_shutdown();
	starpu_mpi_shutdown();
	return ret;
}

/* -----------------------------------------------------------------------
 * Phase 2: restore and verify
 * ----------------------------------------------------------------------- */

/* Restore the counter entry written by old_rank and check its value.
 * Returns the number of errors found. */
static int restore_and_check_counter(int cp_id, int cp_inst, int old_rank, int rank)
{
	int counter_restored = -1;
	int ret = starpu_mpi_checkpoint_restore_value(cp_id, cp_inst,
	                                              old_rank, COUNTER_TAG,
	                                              &counter_restored, sizeof(int));
	if (ret != 0)
	{
		FPRINTF(stderr, "[restore][rank %d] ERROR: restore_value for old rank %d failed (%d)\n",
		        rank, old_rank, ret);
		return 1;
	}

	int expected = 100 + old_rank;
	if (counter_restored != expected)
	{
		FPRINTF(stderr, "[restore][rank %d] ERROR: counter of old rank %d expected %d got %d\n",
		        rank, old_rank, expected, counter_restored);
		return 1;
	}

	FPRINTF(stderr, "[restore][rank %d] counter of old rank %d: OK (%d)\n",
	        rank, old_rank, counter_restored);
	return 0;
}

static int phase_restore(int argc, char *argv[], int rank, int size)
{
	struct starpu_conf conf;
	starpu_init_noworker(&conf);

	int ret = starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_checkpoint_init();

	/* ---- detect checkpoint ---- */
	int restart_flag = 0;
	starpu_mpi_init_from_checkpoint(g_storage_path, &restart_flag);

	if (!restart_flag)
	{
		FPRINTF(stderr, "[restore][rank %d] ERROR: no checkpoint found in %s\n",
		        rank, g_storage_path);
		starpu_mpi_checkpoint_shutdown();
		starpu_mpi_shutdown();
		return 1;
	}

	int cp_id = starpu_mpi_checkpoint_get_restart_cp_id();
	int cp_inst = starpu_mpi_checkpoint_get_restart_cp_inst();
	int old_n = starpu_mpi_checkpoint_get_restart_n_ranks();

	FPRINTF(stderr, "[restore][rank %d] checkpoint found: cp_id=%d cp_inst=%d old_n_ranks=%d\n",
	        rank, cp_id, cp_inst, old_n);

	int errors = 0;

	/* Both phases run in the same job here, so the checkpoint was written by
	 * exactly as many nodes as are running now. */
	if (old_n != size)
	{
		FPRINTF(stderr, "[restore][rank %d] ERROR: manifest reports %d ranks, expected %d\n",
		        rank, old_n, size);
		errors++;
	}

	/* ---- restore velocity handle (rank 0 only) ---- */
	float vel_restored[ARRAY_SIZE];
	memset(vel_restored, 0, sizeof(vel_restored));

	starpu_data_handle_t vel_handle;
	starpu_vector_data_register(&vel_handle, STARPU_MAIN_RAM,
	                            (uintptr_t)vel_restored, ARRAY_SIZE, sizeof(float));
	starpu_mpi_data_register(vel_handle, VEL_TAG, 0);

	if (rank == 0)
	{
		ret = starpu_mpi_checkpoint_restore_handle(cp_id, cp_inst,
		                                           0, VEL_TAG, vel_handle);
		if (ret != 0)
		{
			FPRINTF(stderr, "[restore][rank 0] ERROR: restore_handle failed (%d)\n", ret);
			errors++;
		}
		else
		{
			/* Verify each element */
			starpu_data_acquire(vel_handle, STARPU_R);
			const float *ptr = starpu_data_handle_to_pointer(vel_handle, STARPU_MAIN_RAM);
			for (int i = 0; i < ARRAY_SIZE; i++)
			{
				float expected = (float)(0 * 10 + i);
				if (ptr[i] != expected)
				{
					FPRINTF(stderr,
					        "[restore][rank 0] ERROR: vel[%d] expected %.1f got %.1f\n",
					        i, expected, ptr[i]);
					errors++;
				}
			}
			starpu_data_release(vel_handle);

			if (errors == 0)
				FPRINTF(stderr, "[restore][rank 0] velocity handle: OK\n");
		}
	}

	starpu_data_unregister(vel_handle);

	/* ---- restore the counter this rank wrote ---- */
	int old_rank = (old_n > 0) ? (rank % old_n) : 0;
	errors += restore_and_check_counter(cp_id, cp_inst, old_rank, rank);

	/* ---- restore counters another node wrote ---- */
	/* Entries are addressed by the rank which wrote them precisely so that a
	 * node can take over what a different node owned, which is what happens
	 * when the application resumes on fewer nodes than it was checkpointed
	 * on. That case is reproduced here without needing a second job: pretend
	 * the run resumed on n_new nodes, each of the first n_new ranks taking
	 * over the old ranks congruent to it modulo n_new. The ranks beyond
	 * n_new stand for the nodes which are gone, and restore nothing.
	 *
	 * With the four ranks "make check" uses, rank 0 takes over the old ranks
	 * 0 and 2, and rank 1 the old ranks 1 and 3. */
	if (old_n > 1)
	{
		int n_new = (old_n + 1) / 2;

		if (rank < n_new)
		{
			int claimed;
			for (claimed = rank; claimed < old_n; claimed += n_new)
			{
				if (claimed == old_rank)
					continue; /* already restored above */
				errors += restore_and_check_counter(cp_id, cp_inst, claimed, rank);
			}
		}
	}

	starpu_mpi_checkpoint_shutdown();
	starpu_mpi_shutdown();
	return errors > 0 ? 1 : 0;
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
	int mpi_init;
	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	g_size = size;

	if (size < 2)
	{
		FPRINTF(stderr, "This test requires at least 2 MPI ranks\n");
		MPI_Finalize();
		return 77;
	}

	g_storage_path = storage_path();

	FPRINTF(stderr, "=== Phase 1: write checkpoint ===\n");
	int ret = phase_write(argc, argv, rank, size);

	/* The outcome must be the same on all the ranks: they all take part in
	 * the collective operations below, and a rank leaving on its own would
	 * leave the others waiting. 77 is the skip code, and takes precedence
	 * over the failure code 1. */
	int global_ret = 0;
	MPI_Allreduce(&ret, &global_ret, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (global_ret != 0)
	{
		if (global_ret != 77)
			cleanup_storage(rank);
		if (!mpi_init)
			MPI_Finalize();
		return global_ret;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	FPRINTF(stderr, "=== Phase 2: restore checkpoint ===\n");
	ret = phase_restore(argc, argv, rank, size);
	MPI_Allreduce(&ret, &global_ret, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	cleanup_storage(rank);

	if (rank == 0)
	{
		if (global_ret == 0)
			FPRINTF(stderr, "=== checkpoint_storage test PASSED ===\n");
		else
			FPRINTF(stderr, "=== checkpoint_storage test FAILED ===\n");
	}

	if (!mpi_init)
		MPI_Finalize();
	return global_ret;
}
