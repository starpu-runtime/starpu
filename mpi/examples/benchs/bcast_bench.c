/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021, 2022	    Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Basic broadcast benchmark with synchronized clocks.
 * Inspired a lot from NewMadeleine examples/mcast/nm_mcast_bench.c
 *
 * Synchronized clocks (mpi_sync_clocks) are available here:
 * https://gitlab.inria.fr/pm2/pm2/-/tree/master/mpi_sync_clocks
 * and are detected during StarPU's configure.
 */

#include <starpu_mpi.h>
#include <mpi_sync_clocks.h>
#include "helper.h"
#include "bench_helper.h"

#define SERVER_PRINTF(fmt, ...) do { if(rank == 0) { printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)

typedef void (*algorithm_t)(int nb_dest_nodes, starpu_data_handle_t handle, int nb_nodes_id, int size_id, int bench_id);

static void dummy_loop(int nb_dest_nodes, starpu_data_handle_t handle, int nb_nodes_id, int size_id, int bench_id);

static algorithm_t algorithms[] = { dummy_loop };

#undef NX_MAX
#undef NX_MIN

#define NX_MIN 1

#ifdef STARPU_QUICK_CHECK
#define NB_BENCH 2
#define NX_MAX 100 // kB
#else
#define NB_BENCH 10
#define NX_MAX 240196 // kB
#endif

#define NX_STEP 1.4 // multiplication
#define NB_NODES_STEP 2 // addition
#define NB_NODES_START 3
#define NB_METHODS (sizeof(algorithms)/sizeof(algorithm_t))

struct statistics
{
	double min;
	double med;
	double avg;
	double max;
};

static int times_nb_nodes;
static int times_size;
static int worldsize;
static int rank;
static double* times;
static mpi_sync_clocks_t clocks;

static const starpu_mpi_tag_t data_tag = 0x12;
static const starpu_mpi_tag_t time_tag = 0x13;

static double find_max(double* array, int size)
{
	double t_max = mpi_sync_clocks_remote_to_global(clocks, 1, array[0]);
	double t_value;
	int i;

	for (i = 1; i < size; i++)
	{
		t_value = mpi_sync_clocks_remote_to_global(clocks, i+1, array[i]);
		if (t_value > t_max)
		{
			t_max = t_value;
		}
	}

	return t_max;
}

static struct statistics compute_statistics(double* array, int size)
{
	struct statistics stat;
	int i;

	qsort(array, size, sizeof(double), &comp_double);

	double avg = 0;
	for (i = 0; i < size; i++)
	{
		avg += array[i];
	}
	stat.avg = avg / size;

	stat.min = array[0];
	stat.med = array[(int) floor(size / 2)];
	stat.max = array[size - 1];

	return stat;
}

static int time_index(int size, int bench, int node)
{
	assert(size < times_size);
	assert(bench < NB_BENCH);
	assert(node < worldsize);

	// Warning: if bench < 0 (warmup case), this function returns a result, the user has to check if it makes sense.
	return size * (NB_BENCH * (worldsize + 1)) + bench * (worldsize + 1) + node;
}

static void dummy_loop(int nb_dest_nodes, starpu_data_handle_t data_handle, int nb_nodes_id, int size_id, int bench_id)
{
	double t_end = 0.0;
	int i, ret;
	starpu_data_handle_t time_handle;

	if (rank == 0)
	{
		int t_index = time_index(size_id, bench_id, 0);
		if (bench_id >= 0)
		{
			times[t_index] = mpi_sync_clocks_get_time_usec(clocks);
		}

		starpu_mpi_req* reqs = malloc(nb_dest_nodes*sizeof(starpu_mpi_req));

		for (i = 1; i <= nb_dest_nodes; i++)
		{
			ret = starpu_mpi_isend(data_handle, &reqs[i-1], i, data_tag, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend");
		}

		for (i = 0; i < nb_dest_nodes; i++)
		{
			ret = starpu_mpi_wait(&reqs[i], MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		}

		for (i = 1; i <= nb_dest_nodes; i++)
		{
			starpu_variable_data_register(&time_handle, STARPU_MAIN_RAM, (uintptr_t) &t_end, sizeof(double));
			ret = starpu_mpi_recv(time_handle, i, time_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
			starpu_data_unregister(time_handle);

			if (bench_id >= 0)
			{
				times[t_index+i] = t_end;
			}
		}

		free(reqs);
	}
	else // not server
	{
		ret = starpu_mpi_recv(data_handle, 0, data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		t_end = mpi_sync_clocks_get_time_usec(clocks);

		starpu_variable_data_register(&time_handle, STARPU_MAIN_RAM, (uintptr_t) &t_end, sizeof(double));
		ret = starpu_mpi_send(time_handle, 0, time_tag, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		starpu_data_unregister(time_handle);
	}
}

static void compute_display_times(const int method, const int nb_nodes_id, const int nb_dest_nodes)
{
	int size_id = 0;
	double times_bench[NB_BENCH];
	int s, b;

	SERVER_PRINTF("Computing clock offsets... ");

	mpi_sync_clocks_synchronize(clocks);

	if (rank == 0)
	{
		printf("done\n");

		/* Computing times */
		for (s = NX_MIN; s < NX_MAX; s = (s * NX_STEP) + 1)
		{
			for (b = 0; b < NB_BENCH; b++)
			{
				double t_begin = times[time_index(size_id, b, 0)];
				double t_end = find_max(times + time_index(size_id, b, 1), nb_dest_nodes);
				assert(t_begin < t_end);
				times_bench[b] = t_end - t_begin;
			}

			struct statistics stat_main_task = compute_statistics(times_bench, NB_BENCH);
			printf("   %d    |   %3d  \t| %5d\t\t| ", method, nb_dest_nodes+1, s);
			printf("%10.3lf\t%10.3lf\t%10.3lf\t%10.3lf\n", stat_main_task.min, stat_main_task.med, stat_main_task.avg, stat_main_task.max);
			fflush(stdout);

			size_id++;
		}
	}
}

static inline void man()
{
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-h --help   display this help\n");
	fprintf(stderr, "\t-p          pause workers during benchmark\n");
	exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
	int pause_workers = 0;
	int nb_nodes_id;
	int size_id;
	int thread_support;
	int ret, method, nb_dest_nodes, s, b, i, array_size;
	starpu_data_handle_t data_handle;
	float* msg;

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-p") == 0)
		{
			pause_workers = 1;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			man();
		}
		else
		{
			fprintf(stderr,"Unrecognized option %s\n", argv[i]);
			man();
		}
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

	if (worldsize < 4)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 4 processes.\n");

		starpu_mpi_shutdown();

		return STARPU_TEST_SKIPPED;
	}

	if (pause_workers)
	{
		SERVER_PRINTF("Workers will be paused during benchmark.\n");
		/* Pause workers for this bench: all workers polling for tasks has a strong impact on performances */
		starpu_pause();
	}

	times_nb_nodes = ((worldsize - NB_NODES_START) / NB_NODES_STEP) + 1;
	times_size = (int) (logf((float) NX_MAX / (float) NX_MIN) / logf(NX_STEP)) + 1;
	assert(times_size > 0);

	times = malloc(times_size * NB_BENCH * (worldsize + 1) * sizeof(double));

	SERVER_PRINTF("#0: dummy loop\n");
	SERVER_PRINTF("        |  Nodes  \t|          \t| \tMain task lasted (us):\n");
	SERVER_PRINTF("  Algo  | in comm \t| Size (KB)\t| min\tmed\tavg\tmax\n");
	SERVER_PRINTF("-----------------------------------------------------------------------\n");

	for (method = 0; method < NB_METHODS; method++)
	{
		nb_nodes_id = 0;

		for (nb_dest_nodes = NB_NODES_START; nb_dest_nodes < worldsize; nb_dest_nodes += NB_NODES_STEP)
		{
			starpu_mpi_barrier(MPI_COMM_WORLD);

			SERVER_PRINTF("Starting global clock... ");
			clocks = mpi_sync_clocks_init(MPI_COMM_WORLD);
			SERVER_PRINTF("done\n");

			size_id = 0;

			for (s = NX_MIN; s < NX_MAX; s = (s * NX_STEP) + 1)
			{
				SERVER_PRINTF("   %d    |   %3d  \t| %5d\t\t| ", method, nb_dest_nodes+1, s);

				array_size = s * 1000 / sizeof(float);

				starpu_malloc((void **)&msg, array_size * sizeof(float));
				for (i = 0; i < array_size; i++)
				{
					msg[i] = 3.14;
				}
				starpu_vector_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t) msg, array_size, sizeof(float));

				for (b = -1; b < NB_BENCH; b++)
				{
					if (rank <= nb_dest_nodes)
					{
						algorithms[method](nb_dest_nodes, data_handle, nb_nodes_id, size_id, b);
					}

					SERVER_PRINTF(".");
				}

				SERVER_PRINTF("\n");

				starpu_data_unregister(data_handle);
				starpu_free_noflag(msg, array_size * sizeof(float));
				size_id++;
			}

			// flush clocks
			compute_display_times(method, nb_nodes_id, nb_dest_nodes);
			mpi_sync_clocks_shutdown(clocks);

			nb_nodes_id++;
		}
	}

	if (pause_workers)
	{
		starpu_resume();
	}

	starpu_mpi_shutdown();
	free(times);
	MPI_Finalize();

	return 0;
}
