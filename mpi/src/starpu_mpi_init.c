/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_HAVE_MPI_EXT
#include <mpi-ext.h>
#endif

#ifdef STARPU_SIMGRID
static int _mpi_world_size;
static int _mpi_world_rank;
#endif
static int _mpi_initialized_starpu;
static int _starpu_mpi_gpudirect;	/* Whether GPU direct was explicitly requested (1) or disabled (0), or should be enabled if available (-1) */
int _starpu_mpi_has_cuda;		/* Whether GPU direct is available */
int _starpu_mpi_cuda_devid = -1;	/* Which device GPU direct is enabled for (-1 = all) */

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
#ifdef STARPU_USE_CUDA
	if (_starpu_mpi_gpudirect != 0 && starpu_cuda_worker_get_count() > 0)
	{
		/* Some GPUDirect implementations (e.g. psm2) want cudaSetDevice to be called before MPI_Init */
		int cuda_worker = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
		STARPU_ASSERT(cuda_worker >= 0);
		int devid = starpu_worker_get_devid(cuda_worker);
		STARPU_ASSERT(devid >= 0);
		cudaSetDevice(devid);
	}
#endif
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

	// automatically register the given communicator
	starpu_mpi_comm_register(argc_argv->comm);
	if (argc_argv->comm != MPI_COMM_WORLD)
		starpu_mpi_comm_register(MPI_COMM_WORLD);

	MPI_Comm_rank(argc_argv->comm, &argc_argv->rank);
	MPI_Comm_size(argc_argv->comm, &argc_argv->world_size);
	MPI_Comm_set_errhandler(argc_argv->comm, MPI_ERRORS_RETURN);

#ifdef STARPU_USE_CUDA
#ifdef MPIX_CUDA_AWARE_SUPPORT
	if (MPIX_Query_cuda_support())
		_starpu_mpi_has_cuda = 1;
	else if (_starpu_mpi_gpudirect > 0)
		_STARPU_DISP("Warning: MPI GPUDirect requested, but MPIX_Query_cuda_support reports that it is not supported.");
	_STARPU_DEBUG("MPI has CUDA: %d\n", _starpu_mpi_has_cuda);
	if (!_starpu_mpi_gpudirect)
	{
		_STARPU_DEBUG("But disabled by user\n");
		_starpu_mpi_has_cuda = 0;
	}
	if (_starpu_mpi_has_cuda)
	{
#pragma weak psm2_init
		extern int psm2_init(int *major, int *minor);
		if (psm2_init && starpu_cuda_worker_get_count() > 1)
		{
			int cuda_worker = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
			_starpu_mpi_cuda_devid = starpu_worker_get_devid(cuda_worker);

			_STARPU_DISP("Warning: MPI GPUDirect is enabled using the PSM2 driver, but StarPU will be driving several CUDA GPUs.");
			_STARPU_DISP("Since the PSM2 driver only supports one CUDA GPU at a time for GPU Direct (at least as of its version 11.2.185), StarPU-MPI will use GPU Direct only for CUDA%d.", _starpu_mpi_cuda_devid);
			_STARPU_DISP("To get GPU Direct working with all CUDA GPUs with the PSM2 driver, you will unfortunately have to run one MPI rank per GPU.");
		}
	}
#else
	if (_starpu_mpi_gpudirect > 0)
		_STARPU_DISP("Warning: MPI GPUDirect requested, but the MPIX_Query_cuda_support function is not provided by the MPI Implementation, did you compile it with CUDA support and the Cuda MPI extension?");
	_STARPU_DEBUG("No CUDA support in MPI\n");
#endif
#endif

#ifdef STARPU_SIMGRID
	_mpi_world_size = argc_argv->world_size;
	_mpi_world_rank = argc_argv->rank;
#endif
}

static
void _starpu_mpi_backend_check()
{
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_init != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_shutdown != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_reserve_core != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_request_init != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_request_fill != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_request_destroy != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_data_clear != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_data_register != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_comm_register != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_progress_init != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_progress_shutdown != NULL);
#ifdef STARPU_SIMGRID
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_wait_for_initialization != NULL);
#endif
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_barrier != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_wait_for_all != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_wait != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_test != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_isend_size_func != NULL);
	STARPU_ASSERT(_mpi_backend._starpu_mpi_backend_irecv_size_func != NULL);
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

	_starpu_mpi_backend_check();

	_starpu_mpi_gpudirect = starpu_getenv_number("STARPU_MPI_GPUDIRECT");
#ifdef STARPU_SIMGRID
	/* Call MPI_Init_thread as early as possible, to initialize simgrid
	 * before working with mutexes etc. */
	_starpu_mpi_do_initialize(argc_argv);
#endif

	int ret = _mpi_backend._starpu_mpi_backend_progress_init(argc_argv);

	if (starpu_getenv_number_default("STARPU_DISPLAY_BINDINGS", 0))
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
	_mpi_backend._starpu_mpi_backend_wait_for_initialization();
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
		starpu_mpi_comm_rank(MPI_COMM_WORLD, rank);
		starpu_mpi_comm_size(MPI_COMM_WORLD, world_size);
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
	int mpi_thread_cpuid = starpu_getenv_number_default("STARPU_MPI_THREAD_CPUID", -1);
	int mpi_thread_coreid = starpu_getenv_number_default("STARPU_MPI_THREAD_COREID", -1);
	if (mpi_thread_cpuid < 0 && mpi_thread_coreid < 0 && _mpi_backend._starpu_mpi_backend_reserve_core() && conf->ncpus == -1)
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
	return starpu_mpi_shutdown_comm(MPI_COMM_WORLD);
}

struct comm_size_entry
{
	UT_hash_handle hh;
	MPI_Comm comm;
	int size;
	int rank;
};

static struct comm_size_entry *registered_comms = NULL;

int starpu_mpi_shutdown_comm(MPI_Comm comm)
{
	void *value;
	int rank, world_size;

	/* Make sure we do not have MPI communications pending in the task graph
	 * before shutting down MPI */
	starpu_mpi_wait_for_all(comm);

	/* We need to get the rank before calling MPI_Finalize to pass to _starpu_mpi_comm_amounts_display() */
	starpu_mpi_comm_rank(comm, &rank);
	starpu_mpi_comm_size(comm, &world_size);

	/* kill the progression thread */
	_mpi_backend._starpu_mpi_backend_progress_shutdown(&value);

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

	struct comm_size_entry *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, registered_comms, entry, tmp)
	{
		HASH_DEL(registered_comms, entry);
		free(entry);
	}

	if (_mpi_initialized_starpu)
		starpu_shutdown();

	return 0;
}

int starpu_mpi_comm_register(MPI_Comm comm)
{
	struct comm_size_entry *entry;

	_STARPU_MPI_MALLOC(entry, sizeof(*entry));
	entry->comm = comm;
	MPI_Comm_size(entry->comm, &(entry->size));
	MPI_Comm_rank(entry->comm, &(entry->rank));
	HASH_ADD(hh, registered_comms, comm, sizeof(entry->comm), entry);
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
	struct comm_size_entry *entry;
	HASH_FIND(hh, registered_comms, &comm, sizeof(entry->comm), entry);
	STARPU_ASSERT_MSG(entry, "Communicator %ld has not been registered\n", (long int)comm);
	*size = entry->size;
	return 0;
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
	struct comm_size_entry *entry;
	HASH_FIND(hh, registered_comms, &comm, sizeof(entry->comm), entry);
	STARPU_ASSERT_MSG(entry, "Communicator %ld has not been registered\n", (long int)comm);
	*rank = entry->rank;
	return 0;
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

int starpu_mpi_get_thread_cpuid(void)
{
	return _starpu_mpi_thread_cpuid;
}
