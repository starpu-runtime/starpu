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
 * Inspired a lot from NewMadeleine examples/bench-coll/nm_bench_coll_mcast.c
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

#undef MULT_DEFAULT
#undef LOOPS_DEFAULT

#ifdef STARPU_QUICK_CHECK
#define MIN_DEFAULT   1
#define MAX_DEFAULT   1024
#define LOOPS_DEFAULT 2
#define INCR_DEFAULT  2
#define MULT_DEFAULT  2
#else
#define MIN_DEFAULT   1
#define MAX_DEFAULT   (16*1024*1024)
#define LOOPS_DEFAULT 50
#define INCR_DEFAULT  1
#define MULT_DEFAULT  1.4
#endif

#define NODE_INCREMENT 1

static starpu_data_handle_t data_handle, data_handle_in, data_handle_out;
static int use_tasks = 0;

static void writer_cpu_func(void *descr[], void *args)
{
	(void) descr;
	(void) args;
}

static struct starpu_codelet writer_cl =
{
	.cpu_funcs = { writer_cpu_func },
	.cpu_funcs_name = { "writer_task" },
	.nbuffers = 1,
	.modes = { STARPU_W }
};

static void reader_cpu_func(void* descr[], void* args)
{
	(void) descr;
	(void) args;
}

static struct starpu_codelet reader_cl =
{
	.cpu_funcs = { reader_cpu_func },
	.cpu_funcs_name = { "reader_task" },
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_W }
};

static void usage(void)
{
	fprintf(stderr, "-N iterations - iterations per length [%d]\n", LOOPS_DEFAULT);
	fprintf(stderr, "--tasks - triggers coop through task dependency instead of StarPU's MPI interface\n");
	fprintf(stderr, "-P incr - number of nodes increment [%d]\n", NODE_INCREMENT);
}

static inline uint64_t _next(uint64_t len, double multiplier, uint64_t increment)
{
	uint64_t next = len * multiplier + increment;
	if (next <= len)
		next++;
	return next;
}

static void bcast(MPI_Comm subcomm, int rank, int nb_dests)
{
	int i = 0, ret;

	if (use_tasks)
	{
		starpu_mpi_task_insert(subcomm, &writer_cl, STARPU_W, data_handle_in, 0);
		for (i = 1; i <= nb_dests; i++)
		{
			starpu_mpi_data_register(data_handle_out, i, i);
			starpu_mpi_task_insert(subcomm, &reader_cl, STARPU_R, data_handle_in, STARPU_W, data_handle_out, 0);
		}
		/* Resume StarPU's workers only after submitting tasks, to make
		 * sure the coop will be correctly detected. */
		starpu_resume();
		starpu_task_wait_for_all();

		starpu_pause();
	}
	else
	{
		if (rank == 0)
		{
			/* We explicitely tell StarPU this send will be a broadcast with n recipients. */
			starpu_mpi_coop_sends_data_handle_nb_sends(data_handle, nb_dests);
			for (i = 1; i <= nb_dests; i++)
			{
				ret = starpu_mpi_isend_detached(data_handle, i , 0x42, subcomm, NULL, NULL);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
			}
		}
		else
		{
			ret = starpu_mpi_recv(data_handle, 0, 0x42, subcomm, MPI_STATUS_IGNORE);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		}
	}
}


int main(int argc, char**argv)
{
	const uint64_t start_len  = MIN_DEFAULT;
	const uint64_t end_len    = MAX_DEFAULT;
	const double   multiplier = MULT_DEFAULT;
	const uint64_t increment  = INCR_DEFAULT;
	int iterations            = LOOPS_DEFAULT;
	int node_increment        = NODE_INCREMENT;
	int i, ret, rank, worldsize, subcomm_rank, thread_support;
	MPI_Group world_group;

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-N") == 0)
		{
			iterations = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "--tasks") == 0)
		{
			use_tasks = 1;
		}
		else if (strcmp(argv[i], "-P") == 0)
		{
			node_increment = atoi(argv[++i]);
		}
		else
		{
			fprintf(stderr, "%s: illegal argument %s\n", argv[0], argv[i]);
			usage();
			exit(1);
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

	if (worldsize < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();

		return STARPU_TEST_SKIPPED;
	}

	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	/* Pause workers for this bench, to avoid any impact on performances from polling workers */
	starpu_pause();

	starpu_mpi_barrier(MPI_COMM_WORLD);

	SERVER_PRINTF("# start_len  = %lu B\n", start_len);
	SERVER_PRINTF("# end_len    = %lu B\n", end_len);
	SERVER_PRINTF("# increment  = %lu\n", increment);
	SERVER_PRINTF("# multiplier = %f\n", multiplier);
	SERVER_PRINTF("# iterations = %d\n", iterations);
	SERVER_PRINTF("# coop       = %s\n", starpu_mpi_coop_sends_get_use() ? "on" : "off");
	SERVER_PRINTF("# n.nodes  length        n.iter     min.lat.          median         average        max.lat. \n");

	int nb_nodes;
	for (nb_nodes = 2; nb_nodes <= worldsize; nb_nodes += node_increment)
	{
		SERVER_PRINTF("# starting %d nb_nodes...\n", nb_nodes);

		if (rank >= nb_nodes)
		{
			continue;
		}

		int* group_ranks = malloc(nb_nodes * sizeof(int));
		for (i = 0; i < nb_nodes; i++)
		{
			group_ranks[i] = i;
		}

		MPI_Group sub_group;
		MPI_Group_incl(world_group, nb_nodes, group_ranks, &sub_group);

		MPI_Comm sub_comm;
		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &sub_comm);

		MPI_Comm_rank(sub_comm, &subcomm_rank);

		uint64_t len;
		for (len = start_len; len < end_len; len = _next(len, multiplier, increment))
		{
			char* buf1 = malloc(len);
			char* buf2 = malloc(len);
			/* Precise the buffer where the data will be received, to take benefit from the rcache. */
			if (use_tasks)
			{
				starpu_vector_data_register(&data_handle_in, STARPU_MAIN_RAM, (uintptr_t) buf1, len, 1);
				starpu_vector_data_register(&data_handle_out, STARPU_MAIN_RAM, (uintptr_t) buf2, len, 1);
				starpu_mpi_data_register(data_handle_in, 0, 0);
			}
			else
			{
				starpu_vector_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t) buf1, len, 1);
			}
			mpi_sync_clocks_t clocks = mpi_sync_clocks_init(sub_comm);
			double* lats = (subcomm_rank == 0) ? malloc(iterations * sizeof(double)) : NULL;
			int k;
			for (k = 0; k < iterations; k++)
			{
				int* rc_all = (subcomm_rank == 0) ? malloc(nb_nodes * sizeof(int)) : NULL;
				double local_lat = -1.0;
				int rc = 0;
				do
				{
					const double b = mpi_sync_clocks_barrier(clocks, NULL);
					rc = (b < 0.0);

					const double t_begin = mpi_sync_clocks_get_time_usec(clocks);
					bcast(sub_comm, subcomm_rank, nb_nodes-1);
					const double t_end = mpi_sync_clocks_get_time_usec(clocks);

					local_lat = t_end - t_begin;

					/* collect sync barrier success */
					MPI_Gather(&rc, 1, MPI_INT, rc_all, 1, MPI_INT, 0, sub_comm);
					if (subcomm_rank == 0)
					{
						int i;
						for (i = 0; i < nb_nodes; i++)
						{
							rc |= rc_all[i];
						}
					}
					MPI_Bcast(&rc, 1, MPI_INT, 0, sub_comm);
				} while(rc != 0);

				/* find maximum latency accross nb_nodes */
				double* lat_all = (subcomm_rank == 0) ? malloc(nb_nodes * sizeof(double)) : NULL;
				MPI_Gather(&local_lat, 1, MPI_DOUBLE, lat_all, 1, MPI_DOUBLE, 0, sub_comm);

				if (subcomm_rank == 0)
				{
					int i;
					double max_lat = 0.0;
					for (i = 0; i < nb_nodes; i++)
					{
						if (lat_all[i] > max_lat)
						{
							max_lat = lat_all[i];
						}
					}
					lats[k] = max_lat;
					free(rc_all);
					free(lat_all);
				}
			}

			/* compute time stats accross iterations */
			if (subcomm_rank == 0)
			{
				qsort(lats, iterations, sizeof(double), &comp_double);
				const double min_lat = lats[0];
				const double max_lat = lats[iterations - 1];
				const double med_lat = lats[(iterations - 1) / 2];
				double avg_lat = 0.0;
				for (k = 0; k < iterations; k++)
				{
					avg_lat += lats[k];
				}
				avg_lat /= iterations;
				printf("%4d\t%9lu\t%7d\t%9.3lf\t%9.3lf\t%9.3lf\t%9.3lf \n", nb_nodes, len, iterations, min_lat, med_lat, avg_lat, max_lat);
				fflush(stdout);
				free(lats);
			}
			if (use_tasks)
			{
				starpu_data_unregister(data_handle_in);
				starpu_data_unregister(data_handle_out);
			}
			else
			{
				starpu_data_unregister(data_handle);
			}
			free(buf1);
			free(buf2);
			mpi_sync_clocks_shutdown(clocks);
			clocks = NULL;
		}
	}

	SERVER_PRINTF("# bench end\n");

	MPI_Group_free(&world_group);
	starpu_resume();
	starpu_mpi_shutdown();
	MPI_Finalize();

	return 0;
}
