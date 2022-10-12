/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi_stats.h>
#include <common/config.h>
#include <stdio.h>
#include <starpu_mpi_private.h>
#include <starpu_util.h>

/* measure the amount of data transfers between each pair of MPI nodes */
static size_t *comm_amount;
static int world_size;
static int stats_enabled=0;
static int nb_coop;
static int* nb_nodes_per_coop;
static double time_init;
static MPI_Comm comm_init;
static int nb_sends = 0;
static size_t max_sent_size = 0;
#ifdef STARPU_USE_MPI_NMAD
static struct _starpu_spinlock stats_lock;
#endif

void _starpu_mpi_comm_amounts_init(MPI_Comm comm)
{
	if (stats_enabled != 1)
	{
		time_init = starpu_timing_now();
		comm_init = comm;
		stats_enabled = starpu_getenv_number("STARPU_MPI_STATS");
		if (stats_enabled == -1)
		{
			/* Legacy env var */
			stats_enabled = starpu_getenv_number("STARPU_COMM_STATS");
		}
		if (stats_enabled == -1)
		{
			stats_enabled = 0;
		}
	}
	if (stats_enabled == 0)
		return;

	_STARPU_DISP("Warning: StarPU is executed with STARPU_MPI_STATS=1, which slows down a bit\n");

	starpu_mpi_comm_size(comm, &world_size);
	_STARPU_MPI_DEBUG(1, "allocating for %d nodes\n", world_size);

	_STARPU_MPI_CALLOC(comm_amount, world_size, sizeof(size_t));

	nb_coop = 0;
	_STARPU_MPI_CALLOC(nb_nodes_per_coop, world_size, sizeof(int));

#ifdef STARPU_USE_MPI_NMAD
	_starpu_spin_init(&stats_lock);
#endif
}

void _starpu_mpi_comm_stats_disable()
{
	stats_enabled = 0;
}

void _starpu_mpi_comm_stats_enable()
{
	stats_enabled = 1;
	if (comm_amount == NULL)
	{
		_starpu_mpi_comm_amounts_init(comm_init);
	}
}

void _starpu_mpi_comm_amounts_shutdown()
{
	if (stats_enabled == 0)
		return;
	free(comm_amount);
	free(nb_nodes_per_coop);

#ifdef STARPU_USE_MPI_NMAD
	_starpu_spin_destroy(&stats_lock);
#endif
}

void _starpu_mpi_comm_amounts_inc(MPI_Comm comm, unsigned dst, MPI_Datatype datatype, int count)
{
	int src, size;

	if (stats_enabled == 0)
		return;

	starpu_mpi_comm_rank(comm, &src);
	MPI_Type_size(datatype, &size);

	_STARPU_MPI_DEBUG(1, "[%d] adding %d to %d\n", src, count*size, dst);

#ifdef STARPU_USE_MPI_NMAD
	/* With NewMadeleine, the send requests are triggered from the workers, so
	 * this is a critical section. */
	_starpu_spin_lock(&stats_lock);
#endif

	comm_amount[dst] += count*size;

	if (((size_t) count*size) > max_sent_size)
	{
		max_sent_size = count*size;
	}

	nb_sends++;

#ifdef STARPU_USE_MPI_NMAD
	_starpu_spin_unlock(&stats_lock);
#endif
}

void _starpu_mpi_nb_coop_inc(int nb_nodes_in_coop)
{
	if (stats_enabled == 0)
		return;

	assert(nb_nodes_in_coop > 0);
	assert(nb_nodes_in_coop < world_size);

#ifdef STARPU_USE_MPI_NMAD
	STARPU_ATTRIBUTE_UNUSED size_t dummy = STARPU_ATOMIC_ADD(&nb_coop, 1);
	dummy = STARPU_ATOMIC_ADD(&nb_nodes_per_coop[nb_nodes_in_coop-1], 1);
#else
	nb_coop++;
	nb_nodes_per_coop[nb_nodes_in_coop-1]++;
#endif
}

void starpu_mpi_comm_amounts_retrieve(size_t *comm_amounts)
{
	if (stats_enabled == 0)
		return;
	memcpy(comm_amounts, comm_amount, world_size * sizeof(size_t));
}

void _starpu_mpi_comm_amounts_display(FILE *stream, int node)
{
	int dst;
	size_t sum = 0;

	if (comm_amount == NULL)
		return;

	double time = starpu_timing_now() - time_init;

	for (dst = 0; dst < world_size; dst++)
	{
		sum += comm_amount[dst];
	}

	fprintf(stream, "\n[starpu_comm_stats][%d] TOTAL:\t%f B\t%f MB\t %f B/s\t %f MB/s\n", node, (float)sum, (float)sum/1024/1024, (float)sum/(float)time, (float)sum/1204/1024/(float)time);

	fprintf(stream, "[starpu_comm_stats][%d] nb_sends: %d\n", node, nb_sends);
	fprintf(stream, "[starpu_comm_stats][%d] max_sent_size: %ld\n", node, max_sent_size);
	fprintf(stream, "[starpu_comm_stats][%d] average sent size: %ld\n", node, nb_sends ? sum / nb_sends : 0);

	for (dst = 0; dst < world_size; dst++)
	{
		if (comm_amount[dst])
			fprintf(stream, "[starpu_comm_stats][%d:%d]\t%f B\t%f MB\t %f B/s\t %f MB/s\n",
				node, dst, (float)comm_amount[dst], ((float)comm_amount[dst])/(1024*1024),
				(float)comm_amount[dst]/(float)time, ((float)comm_amount[dst])/(1024*1024)/(float)time);
	}

	fprintf(stream, "[starpu_comm_stats][%d] NB_COOP: %d\n", node, nb_coop);
	for (dst = 0; dst < world_size; dst++)
	{
		if (nb_nodes_per_coop[dst] != 0)
		{
			fprintf(stream, "[starpu_comm_stats][%d]\t %d in coop: %d (%f%%)\n", node, dst+1, nb_nodes_per_coop[dst], nb_coop ? (((float) nb_nodes_per_coop[dst]) / nb_coop) * 100. : 0);
		}
	}
}
