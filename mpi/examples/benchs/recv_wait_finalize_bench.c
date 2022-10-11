/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This benchmark measures the impact of the STARPU_MPI_RECV_FINALIZE env var:
 * when set to 0, StarPU can use received buffers for task only reading these
 * buffers, while the communication library still holds a reference on this
 * buffer (to continue the tree broadcast, for instance).
 * Inspired a lot from NewMadeleine examples/mcast/nm_mcast_prio.c
 *
 * Synchronized clocks (mpi_sync_clocks) are available here:
 * https://gitlab.inria.fr/pm2/pm2/-/tree/master/mpi_sync_clocks
 * and are detected during StarPU's configure.
 */

#include <starpu_mpi.h>
#include <mpi_sync_clocks.h>
#include "helper.h"

#define SERVER_PRINTF(fmt, ...) do { if(rank == 0) { printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)

#define DEFAULT_ARRAY_SIZE 1
#ifdef STARPU_QUICK_CHECK
  #define DEFAULT_ROUND 5
#else
  #define DEFAULT_ROUND 200
#endif

static starpu_data_handle_t data_handle;
static int rank;
static double received_time, finalized_time;
static mpi_sync_clocks_t clocks;
static int* prios;

// Codelet executed just to block start of the broadcast and be sure the broadcast will be correctly detected:
static void trigger_coop_cpu_func(void *descr[], void *args)
{
	(void) descr;
	(void) args;
}

static struct starpu_codelet trigger_coop_cl =
{
	.cpu_funcs = { trigger_coop_cpu_func },
	.cpu_funcs_name = { "trigger_coop_task" },
	.name = "trigger_coop",
	.nbuffers = 1,
	.modes = { STARPU_W }
};

// Codelet executed when data just arrived, but communication library has still a reference on it
static void received_cpu_func(void *descr[], void *args)
{
	(void) descr;
	(void) args;

	received_time = mpi_sync_clocks_get_time_usec(clocks);
}

static struct starpu_codelet received_cl =
{
	.cpu_funcs = { received_cpu_func },
	.cpu_funcs_name = { "received_task" },
	.name = "received",
	.nbuffers = 1,
	.modes = { STARPU_R }
};

// Codelet executed when data is released by communication library
static void finalized_cpu_func(void *descr[], void *args)
{
	(void) descr;
	(void) args;

	finalized_time = mpi_sync_clocks_get_time_usec(clocks);
}

static struct starpu_codelet finalized_cl =
{
	.cpu_funcs = { finalized_cpu_func },
	.cpu_funcs_name = { "finalized_task" },
	.name = "finalized",
	.nbuffers = 1,
	.modes = { STARPU_W }
};


static void usage(void)
{
	fprintf(stderr, "-s array size - number of bytes to broadcast [%d]\n", DEFAULT_ARRAY_SIZE);
	fprintf(stderr, "-rounds rounds - number of iterations [%d]\n", DEFAULT_ROUND);
}


static void bcast(int nb_dests, double* time_to_receive, double* time_to_finalize)
{
	int i = 0;

	starpu_mpi_data_set_rank(data_handle, 0);

	/* This first task is just to retain communications, and be sure they
	 * will be detected as a broadcast, if there are enough nodes. */
	starpu_mpi_task_insert(MPI_COMM_WORLD, &trigger_coop_cl, STARPU_W, data_handle, 0);

	for (i = 1; i < nb_dests; i++)
	{
		starpu_mpi_task_insert(MPI_COMM_WORLD, &received_cl,
				       STARPU_R, data_handle,
				       STARPU_EXECUTE_ON_NODE, i,
				       STARPU_PRIORITY, prios[i-1],
				       0);
	}
	for (i = 1; i < nb_dests; i++)
	{
		/* Little bit hacky here: we change the owner of the handle to
		 * be the node on which we are just about to submit a task to
		 * be executed on that node, with this handle. This is done to
		 * avoid additional communications we don't want in this bench.
		 * In real applications, the coherency of the data will
		 * probably be broken, but for this bench we don't care. */
		starpu_mpi_data_set_rank(data_handle, i);
		starpu_mpi_task_insert(MPI_COMM_WORLD, &finalized_cl, STARPU_W, data_handle, 0);
	}

	mpi_sync_clocks_barrier(clocks, NULL);

	const double t_begin = mpi_sync_clocks_get_time_usec(clocks);

	/* Resume StarPU's workers only after submitting tasks, to make
	  * sure the coop will be correctly detected. */
	starpu_resume();
	starpu_task_wait_for_all();

	starpu_pause();

	*time_to_receive = received_time - t_begin;
	*time_to_finalize = finalized_time - t_begin;
}


int main(int argc, char**argv)
{
	int i, ret, worldsize, rounds = DEFAULT_ROUND, thread_support;
	long long int s = DEFAULT_ARRAY_SIZE;
	double time_to_receive, time_to_finalize;
	double total_time_to_receive = 0.0, total_time_to_finalize = 0.0;

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-s") == 0)
		{
			s = (int long long) atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "-rounds") == 0)
		{
			rounds = atoi(argv[++i]);
			continue;
		}
		else
		{
			fprintf(stderr, "%s: illegal argument %s\n", argv[0], argv[i]);
			usage();
			exit(1);
		}
	}

	if (rounds <= 0)
	{
		FPRINTF(stderr, "The number of iterations has to be greater than 0.\n");
		return EXIT_FAILURE;
	}

	if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support) != MPI_SUCCESS)
	{
		FPRINTF(stderr, "MPI_Init_thread failed\n");
		return EXIT_FAILURE;
	}

	if (thread_support < MPI_THREAD_MULTIPLE)
	{
		/* We need MPI_THREAD_MULTIPLE for the StarPU's MPI thread and
		 * the main thread calling functions from mpi_sync_clocks. */
		FPRINTF(stderr, "This benchmark requires MPI_THREAD_MULTIPLE support.\n");
		MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

        ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	if (worldsize < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();

		return STARPU_TEST_SKIPPED;
	}

	/* Pause workers for this bench, to avoid any impact on performances
	 * from polling workers, and to detect correctly coop */
	starpu_pause();

	starpu_mpi_barrier(MPI_COMM_WORLD);

	SERVER_PRINTF("# message size = %lld B\n", s);
	SERVER_PRINTF("# iterations   = %d\n", rounds);
	SERVER_PRINTF("# coop         = %s\n", starpu_mpi_coop_sends_get_use() ? "on" : "off");
	SERVER_PRINTF("# node ; prio ; delay data (usec.); finalized (usec.)\n");

	clocks = mpi_sync_clocks_init(MPI_COMM_WORLD);

	prios = malloc((worldsize-1) * sizeof(int));
	for (i = 0; i < worldsize-1; i++)
	{
		prios[i] = i;
	}

	char* buffer = malloc(s);
	memset(buffer, 0, s);

	/* To keep the same buffer and get good performances with rcache, we
	 * provide the buffer for sender and receivers. If we let StarPU manage
	 * the buffer, it can change it between iterations.
	 * The original owner of the data (the sender) is defined with
	 * starpu_mpi_data_set_rank() in bcast(). */
	starpu_vector_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t) buffer, s, sizeof(char));
	starpu_mpi_data_set_tag(data_handle, 0xee);

	for (i = 0; i < rounds; i++)
	{
		bcast(worldsize, &time_to_receive, &time_to_finalize);
		total_time_to_receive += time_to_receive;
		total_time_to_finalize += time_to_finalize;
	}

	total_time_to_receive /= rounds;
	total_time_to_finalize /= rounds;

	if (rank == 0)
	{
		double* totals_time_to_receive = malloc(sizeof(double) * worldsize);
		double* totals_time_to_finalize = malloc(sizeof(double) * worldsize);

		MPI_Gather(&total_time_to_receive, 1, MPI_DOUBLE, totals_time_to_receive, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(&total_time_to_finalize, 1, MPI_DOUBLE, totals_time_to_finalize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		for (i = 1; i < worldsize; i++)
		{
			printf("%d \t %d \t %g \t %g\n", i, prios[i-1], totals_time_to_receive[i], totals_time_to_finalize[i]);
		}

		free(totals_time_to_receive);
		free(totals_time_to_finalize);
	}
	else
	{
		MPI_Gather(&total_time_to_receive, 1, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(&total_time_to_finalize, 1, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	starpu_data_unregister(data_handle);
	free(buffer);

	free(prios);

	mpi_sync_clocks_shutdown(clocks);

	SERVER_PRINTF("# bench end\n");

	starpu_resume();
	starpu_mpi_shutdown();
	MPI_Finalize();

	return 0;
}
