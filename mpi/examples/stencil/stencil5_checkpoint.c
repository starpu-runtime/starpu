/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2026-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi.h>
#include "helper.h"

/*
 * A one-dimensional stencil that can be resumed from a checkpoint written to a
 * shared filesystem, showing how little an iterative application has to do to
 * survive being interrupted.
 *
 * Each cell holds one float and is updated from its two neighbours. The cells
 * are distributed over the nodes in contiguous blocks. Every -period
 * iterations, the cells a node owns and the iteration counter are written to
 * the directory given by -path.
 *
 * Run it once, interrupt it, and run it again with the same -path: it reports
 * the checkpoint it found, resumes from there, and reaches the value an
 * uninterrupted run would have reached.
 *
 *   mpirun -np 2 ./stencil5_checkpoint -path /shared/cp -iter 20 -period 5
 */

#define N       32              /* number of cells */
#define ITER_TAG (N + 1)        /* message tag of the iteration counter */
#define CP_ID    1

static int niter = 20;
static int period = 5;
static const char *cp_path = NULL;

/* Which node owns cell i: contiguous blocks. */
static int cell_owner(int i, int nb_nodes)
{
	return (i * nb_nodes) / N;
}

/* Which node keeps the backup of what rank owns. */
static int backup_of(int rank)
{
	int size;
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
	return (rank + 1) % size;
}

void stencil_cpu(void *descr[], void *_args)
{
	(void)_args;
	float *c = (float *)STARPU_VARIABLE_GET_PTR(descr[0]);
	float *l = (float *)STARPU_VARIABLE_GET_PTR(descr[1]);
	float *r = (float *)STARPU_VARIABLE_GET_PTR(descr[2]);

	*c = (*l + *c + *r) / 3;
}

struct starpu_codelet stencil_cl =
{
	.cpu_funcs = {stencil_cpu},
	.nbuffers = 3,
	.modes = {STARPU_RW, STARPU_R, STARPU_R},
	.model = &starpu_perfmodel_nop,
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-iter") == 0)
			niter = atoi(argv[++i]);
		else if (strcmp(argv[i], "-period") == 0)
			period = atoi(argv[++i]);
		else if (strcmp(argv[i], "-path") == 0)
			cp_path = argv[++i];
	}
}

int main(int argc, char **argv)
{
	int my_rank, size, i, loop, ret;
	int first_iter = 0;
	float cells[N];
	starpu_data_handle_t handles[N];
	starpu_mpi_checkpoint_template_t cp_template = NULL;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return my_rank == 0 ? 77 : 0;
	}

	parse_args(argc, argv);
	starpu_mpi_checkpoint_init();

	/* Is there a checkpoint to resume from? This has to be asked before the
	 * template is registered, and it also configures the storage path. */
	int restart = 0;
	int cp_id = -1, cp_inst = -1, old_n = -1;
	int resume_at = 0;	/* the iteration a resumed run has to start from */

	if (cp_path)
	{
		starpu_mpi_init_from_checkpoint(cp_path, &restart);
		if (restart)
		{
			cp_id = starpu_mpi_checkpoint_get_restart_cp_id();
			cp_inst = starpu_mpi_checkpoint_get_restart_cp_inst();
			old_n = starpu_mpi_checkpoint_get_restart_n_ranks();
			FPRINTF_MPI(stderr, "resuming from checkpoint %d/%d written by %d node(s)\n",
				    cp_id, cp_inst, old_n);
		}
		else
		{
			FPRINTF_MPI(stderr, "no checkpoint found, starting from scratch\n");
		}
	}

	/* A step, which the stencil smooths out. A ramp would not do: the average
	 * of three consecutive values of a straight line is the middle one, so a
	 * ramp is a fixed point and nothing would ever change.
	 * Overwritten below if we are resuming. */
	for (i = 0; i < N; i++)
		cells[i] = (i < N/2) ? 0.f : 100.f;

	/* Register the cells this node owns, and the two it needs from its
	 * neighbours */
	for (i = 0; i < N; i++)
	{
		int owner = cell_owner(i, size);
		if (owner == my_rank)
			starpu_variable_data_register(&handles[i], STARPU_MAIN_RAM,
						      (uintptr_t)&cells[i], sizeof(float));
		else if ((i > 0 && cell_owner(i-1, size) == my_rank)
			 || (i < N-1 && cell_owner(i+1, size) == my_rank))
			starpu_variable_data_register(&handles[i], -1, (uintptr_t)NULL, sizeof(float));
		else
		{
			handles[i] = NULL;
			continue;
		}
		starpu_mpi_data_register(handles[i], i, owner);
	}

	/* Restore what the previous run left. Cells are named by the node that
	 * wrote them, so a node reads the entries of the old node whose cells it
	 * now owns; here the distribution is the same, so that is itself. */
	if (restart)
	{
		for (i = 0; i < N; i++)
		{
			if (cell_owner(i, size) != my_rank)
				continue;
			ret = starpu_mpi_checkpoint_restore_handle(cp_id, cp_inst, my_rank, i, handles[i]);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_checkpoint_restore_handle");
		}
		ret = starpu_mpi_checkpoint_restore_value(cp_id, cp_inst, 0, ITER_TAG,
							  &resume_at, sizeof(resume_at));
		first_iter = resume_at;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_checkpoint_restore_value");
		FPRINTF_MPI(stderr, "restored, resuming at iteration %d\n", first_iter);
	}

	/* What a checkpoint holds: the cells this node owns, and the iteration
	 * counter. The counter is the same on every node, so only node 0 needs to
	 * write it. */
	if (cp_path)
	{
		starpu_mpi_checkpoint_template_create(&cp_template, CP_ID, 0);
		for (i = 0; i < N; i++)
			if (cell_owner(i, size) == my_rank)
				starpu_mpi_checkpoint_template_add_entry(&cp_template, STARPU_R,
									 handles[i], backup_of(my_rank), 0);
		if (my_rank == 0)
			starpu_mpi_checkpoint_template_add_entry(&cp_template, STARPU_VALUE,
								 &resume_at, sizeof(resume_at), ITER_TAG, backup_of, 0);
		starpu_mpi_checkpoint_template_freeze(&cp_template);
	}

	for (loop = first_iter; loop < niter; loop++)
	{
		starpu_iteration_push(loop);

		for (i = 1; i < N-1; i++)
			starpu_mpi_task_insert(MPI_COMM_WORLD, &stencil_cl,
					       STARPU_RW, handles[i],
					       STARPU_R, handles[i-1],
					       STARPU_R, handles[i+1],
					       0);

		starpu_iteration_pop();

		if (cp_template && (loop + 1) % period == 0)
		{
			/* The cells are being written by the tasks above, so the
			 * checkpoint is taken at a quiescent point. An application
			 * whose checkpointed data is not under modification does not
			 * need this wait. */
			starpu_task_wait_for_all();
			/* iteration loop is done, so a resumed run starts at the next one */
			resume_at = loop + 1;
			ret = starpu_mpi_checkpoint_flush_to_storage(cp_template);
			if (ret == 0)
			{
				FPRINTF_MPI(stderr, "checkpoint written after iteration %d\n", loop);
			}
			else
			{
				FPRINTF_MPI(stderr, "checkpoint after iteration %d was discarded\n", loop);
			}
		}
	}
	starpu_task_wait_for_all();

	for (i = 0; i < N; i++)
		if (handles[i])
			starpu_data_unregister(handles[i]);

	if (my_rank == 0)
	{
		/* The stencil conserves the mean, so the spread is what shows how far
		 * the computation got: it decreases as the values are smoothed. */
		float mean = 0, spread = 0;
		for (i = 0; i < N; i++)
			mean += cells[i];
		mean /= N;
		for (i = 0; i < N; i++)
			spread += (cells[i] - mean) * (cells[i] - mean);
		FPRINTF(stderr, "after %d iterations: mean %2.4f spread %2.4f\n",
			niter, mean, spread / N);
	}

	starpu_mpi_checkpoint_shutdown();
	starpu_mpi_shutdown();
	return 0;
}
