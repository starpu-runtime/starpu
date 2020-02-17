/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <starpu_profiling.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_cache.h>
#include <starpu_mpi_select_node.h>
#include <common/config.h>
#include <common/thread.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <core/simgrid.h>
#include <core/task.h>

#ifdef STARPU_SIMGRID
static int _mpi_world_size;
static int _mpi_world_rank;
#endif
static int _mpi_initialized_starpu;

static void _starpu_mpi_print_thread_level_support(int thread_level, char *msg)
{
	switch (thread_level)
	{
		case MPI_THREAD_SERIALIZED:
		{
			_STARPU_DISP("MPI%s MPI_THREAD_SERIALIZED; Multiple threads may make MPI calls, but only one at a time.\n", msg);
			break;
		}
		case MPI_THREAD_FUNNELED:
		{
			_STARPU_DISP("MPI%s MPI_THREAD_FUNNELED; The application can safely make calls to StarPU-MPI functions, but should not call directly MPI communication functions.\n", msg);
			break;
		}
		case MPI_THREAD_SINGLE:
		{
			_STARPU_DISP("MPI%s MPI_THREAD_SINGLE; MPI does not have multi-thread support, this might cause problems. The application can make calls to StarPU-MPI functions, but not call directly MPI Communication functions.\n", msg);
			break;
		}
		case MPI_THREAD_MULTIPLE:
			/* no problem */
			break;
	}
}

void _starpu_mpi_do_initialize(struct _starpu_mpi_argc_argv *argc_argv)
{
	if (argc_argv->initialize_mpi)
	{
		STARPU_ASSERT_MSG(argc_argv->comm == MPI_COMM_WORLD, "It does not make sense to ask StarPU-MPI to initialize MPI while a non-world communicator was given");
		int thread_support;
		_STARPU_DEBUG("Calling MPI_Init_thread\n");
		if (MPI_Init_thread(argc_argv->argc, argc_argv->argv, MPI_THREAD_SERIALIZED, &thread_support) != MPI_SUCCESS)
		{
			_STARPU_ERROR("MPI_Init_thread failed\n");
		}
		_starpu_mpi_print_thread_level_support(thread_support, "_Init_thread level =");
	}
	else
	{
		int provided;
		MPI_Query_thread(&provided);
		_starpu_mpi_print_thread_level_support(provided, " has been initialized with");
	}

	MPI_Comm_rank(argc_argv->comm, &argc_argv->rank);
	MPI_Comm_size(argc_argv->comm, &argc_argv->world_size);
	MPI_Comm_set_errhandler(argc_argv->comm, MPI_ERRORS_RETURN);

#ifdef STARPU_SIMGRID
	_mpi_world_size = argc_argv->world_size;
	_mpi_world_rank = argc_argv->rank;
#endif
}

static
int _starpu_mpi_initialize(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm)
{
	struct _starpu_mpi_argc_argv *argc_argv;
	_STARPU_MALLOC(argc_argv, sizeof(struct _starpu_mpi_argc_argv));
	argc_argv->initialize_mpi = initialize_mpi;
	argc_argv->argc = argc;
	argc_argv->argv = argv;
	argc_argv->comm = comm;
	_starpu_implicit_data_deps_write_hook(_starpu_mpi_data_flush);

#ifdef STARPU_SIMGRID
	/* Call MPI_Init_thread as early as possible, to initialize simgrid
	 * before working with mutexes etc. */
	_starpu_mpi_do_initialize(argc_argv);
#endif

	int ret = _starpu_mpi_progress_init(argc_argv);

	if (starpu_get_env_number_default("STARPU_DISPLAY_BINDINGS", 0))
	{
		int rank, size, i;
		char hostname[65];

		starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
		starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
		gethostname(hostname, sizeof(hostname));

		/* We make a barrier between each node calling hwloc-ps, to avoid mixing
		 * outputs in stdout. */
		for (i = 0; i < size; i++)
		{
			starpu_mpi_barrier(MPI_COMM_WORLD);
			if (rank == i)
			{
				fprintf(stdout, "== Binding for rank %d on node %s ==\n", rank, hostname);
				starpu_display_bindings();
				fflush(stdout);
			}
		}
		starpu_mpi_barrier(MPI_COMM_WORLD);
		if (rank == 0)
		{
			fprintf(stdout, "== End of bindings ==\n");
			fflush(stdout);
		}
	}

	return ret;
}

#ifdef STARPU_SIMGRID
/* This is called before application's main, to initialize SMPI before we can
 * create MSG processes to run application's main */
int _starpu_mpi_simgrid_init(int argc, char *argv[])
{
	return _starpu_mpi_initialize(&argc, &argv, 1, MPI_COMM_WORLD);
}
#endif

int starpu_mpi_init_comm(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm)
{
#ifdef STARPU_SIMGRID
	(void)argc;
	(void)argv;
	(void)initialize_mpi;
	(void)comm;
	_starpu_mpi_wait_for_initialization();
	return 0;
#else
	return _starpu_mpi_initialize(argc, argv, initialize_mpi, comm);
#endif
}

int starpu_mpi_init(int *argc, char ***argv, int initialize_mpi)
{
	return starpu_mpi_init_comm(argc, argv, initialize_mpi, MPI_COMM_WORLD);
}

int starpu_mpi_initialize(void)
{
#ifdef STARPU_SIMGRID
	return 0;
#else
	return _starpu_mpi_initialize(NULL, NULL, 0, MPI_COMM_WORLD);
#endif
}

int starpu_mpi_initialize_extended(int *rank, int *world_size)
{
#ifdef STARPU_SIMGRID
	*world_size = _mpi_world_size;
	*rank = _mpi_world_rank;
	return 0;
#else
	int ret;

	ret = _starpu_mpi_initialize(NULL, NULL, 1, MPI_COMM_WORLD);
	if (ret == 0)
	{
		_STARPU_DEBUG("Calling MPI_Comm_rank\n");
		MPI_Comm_rank(MPI_COMM_WORLD, rank);
		MPI_Comm_size(MPI_COMM_WORLD, world_size);
	}
	return ret;
#endif
}

int starpu_mpi_init_conf(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm, struct starpu_conf *conf)
{
	struct starpu_conf localconf;
	if (!conf)
	{
		starpu_conf_init(&localconf);
		conf = &localconf;
	}

	_mpi_backend._starpu_mpi_backend_init(conf);

	/* Reserve a core only if required by the backend and if STARPU_NCPU isn't provided */
	if (_mpi_backend._starpu_mpi_backend_reserve_core() && conf->ncpus == -1)
	{
		/* Reserve a core for our progression thread */
		if (conf->reserve_ncpus == -1)
			conf->reserve_ncpus = 1;
		else
			conf->reserve_ncpus++;
	}

	conf->will_use_mpi = 1;

	int ret = starpu_init(conf);
	if (ret < 0)
		return ret;
	_mpi_initialized_starpu = 1;

	return starpu_mpi_init_comm(argc, argv, initialize_mpi, comm);
}

int starpu_mpi_shutdown(void)
{
	void *value;
	int rank, world_size;

	/* Make sure we do not have MPI communications pending in the task graph
	 * before shutting down MPI */
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	/* We need to get the rank before calling MPI_Finalize to pass to _starpu_mpi_comm_amounts_display() */
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &world_size);

	/* kill the progression thread */
	_starpu_mpi_progress_shutdown(&value);

#ifdef STARPU_USE_FXT
	if (starpu_fxt_is_enabled())
	{
		_STARPU_MPI_TRACE_STOP(rank, world_size);
	}
#endif // STARPU_USE_FXT

	_starpu_mpi_comm_amounts_display(stderr, rank);
	_starpu_mpi_comm_amounts_shutdown();
	_starpu_mpi_cache_shutdown(world_size);

	_mpi_backend._starpu_mpi_backend_shutdown();

	if (_mpi_initialized_starpu)
		starpu_shutdown();

	return 0;
}

int starpu_mpi_comm_size(MPI_Comm comm, int *size)
{
	if (_starpu_mpi_fake_world_size != -1)
	{
		*size = _starpu_mpi_fake_world_size;
		return 0;
	}
#ifdef STARPU_SIMGRID
	STARPU_MPI_ASSERT_MSG(comm == MPI_COMM_WORLD, "StarPU-SMPI only works with MPI_COMM_WORLD for now");
	*size = _mpi_world_size;
	return 0;
#else
	return MPI_Comm_size(comm, size);
#endif
}

int starpu_mpi_comm_rank(MPI_Comm comm, int *rank)
{
	if (_starpu_mpi_fake_world_rank != -1)
	{
		*rank = _starpu_mpi_fake_world_rank;
		return 0;
	}
#ifdef STARPU_SIMGRID
	STARPU_MPI_ASSERT_MSG(comm == MPI_COMM_WORLD, "StarPU-SMPI only works with MPI_COMM_WORLD for now");
	*rank = _mpi_world_rank;
	return 0;
#else
	return MPI_Comm_rank(comm, rank);
#endif
}

int starpu_mpi_world_size(void)
{
	int size;
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
	return size;
}

int starpu_mpi_world_rank(void)
{
	int rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	return rank;
}
