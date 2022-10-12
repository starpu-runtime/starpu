/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021, 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi_private.h>
#include <starpu_mpi_fxt.h>

#ifdef STARPU_HAVE_MPI_SYNC_CLOCKS
#include <mpi_sync_clocks.h>

static mpi_sync_clocks_t mpi_sync_clock;
#endif

static int fxt_random_number = -1;

#if defined(STARPU_HAVE_MPI_SYNC_CLOCKS) && !defined(STARPU_SIMGRID)
/* Use the same clock as the one used by mpi_sync_clocks */
uint64_t fut_getstamp(void)
{
	sync_clocks_generic_tick_t tick;
	sync_clocks_generic_get_tick(tick);
	return (uint64_t) (sync_clocks_generic_tick2usec(tick)*1000.);
}
#endif

static void _starpu_mpi_add_sync_point_in_fxt(void)
{
	int rank, worldsize, ret;

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	STARPU_ASSERT(worldsize > 1);

	ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier returning %s", _starpu_mpi_get_mpi_error_code(ret));

	if (fxt_random_number == -1) // only for the first sync point
	{
		/* We generate a "unique" key so that we can make sure that different
		* FxT traces come from the same MPI run. */
		if (rank == 0)
			fxt_random_number = time(NULL);

		_STARPU_MPI_DEBUG(3, "unique key %x\n", fxt_random_number);

		ret = MPI_Bcast(&fxt_random_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
		STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Bcast returning %s", _starpu_mpi_get_mpi_error_code(ret));
	}

#ifdef STARPU_HAVE_MPI_SYNC_CLOCKS
	if (starpu_getenv_number("STARPU_MPI_TRACE_SYNC_CLOCKS") != 0)
	{
		mpi_sync_clocks_synchronize(mpi_sync_clock);
		double local_sync_time;
		mpi_sync_clocks_barrier(mpi_sync_clock, &local_sync_time);
		/* Even if with this synchronized barrier, all nodes are supposed to left
		* out the barrier exactly at the same time, we can't be sure, the
		* following event will be recorded at the same time on each MPI processes,
		* because this thread can be preempted between the end of the barrier and
		* the event record. That's why we need to store the local time when the
		* barrier was unlocked as an additional information of the event, we can't
		* rely on the timestamp of the event. */
		_STARPU_MPI_TRACE_BARRIER(rank, worldsize, fxt_random_number, (mpi_sync_clocks_get_time_origin_usec(mpi_sync_clock) + local_sync_time) * 1000.);
	}
	else /* mpi_sync_synchronize() can be long (several seconds), one can prefer to use a less precise but faster method: */
#endif
	{
		ret = MPI_Barrier(MPI_COMM_WORLD);
		STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier returning %s", _starpu_mpi_get_mpi_error_code(ret));

		_STARPU_MPI_TRACE_BARRIER(rank, worldsize, fxt_random_number, 0);
	}
}

void _starpu_mpi_fxt_init(void* arg)
{
	struct _starpu_mpi_argc_argv *argc_argv = (struct _starpu_mpi_argc_argv *) arg;

	if (_starpu_fxt_wait_initialisation())
	{
#ifdef STARPU_HAVE_MPI_SYNC_CLOCKS
		if (argc_argv->world_size > 1 && starpu_getenv_number("STARPU_MPI_TRACE_SYNC_CLOCKS") != 0)
		{
			mpi_sync_clock = mpi_sync_clocks_init(MPI_COMM_WORLD);
		}
#endif

		/* We need to record our ID in the trace before the main thread makes any MPI call */
		_STARPU_MPI_TRACE_START(argc_argv->rank, argc_argv->world_size);
		starpu_profiling_set_id(argc_argv->rank);
		_starpu_profiling_set_mpi_worldsize(argc_argv->world_size);

		if (argc_argv->world_size > 1)
		{
			_starpu_mpi_add_sync_point_in_fxt();
		}
	}
}

void _starpu_mpi_fxt_shutdown()
{
	if (starpu_fxt_is_enabled())
	{
		int worldsize;
		starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

		if (worldsize > 1)
		{
			/* We add a synchronization point at the end of the trace,
			* to be able to interpolate times, in order to correct
			* time drift.
			*/
			_starpu_mpi_add_sync_point_in_fxt();

#ifdef STARPU_HAVE_MPI_SYNC_CLOCKS
			if (starpu_getenv_number("STARPU_MPI_TRACE_SYNC_CLOCKS") != 0)
			{
				mpi_sync_clocks_shutdown(mpi_sync_clock);
			}
#endif
		}
	}
}
