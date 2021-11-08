/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <unistd.h>
#include "helper.h"

#ifdef STARPU_QUICK_CHECK
#  define DEFAULT_NITER 16
#else
#  define DEFAULT_NITER 2048
#endif

#define DEFAULT_DATA_SIZE 16
#define DEFAULT_SLEEP_TIME 0
#define DEFAULT_METHOD 0 // ping pongs

void usage()
{
	fprintf(stderr, "-n [number of iteration] (default: %d)\n", DEFAULT_NITER);
	fprintf(stderr, "-s [number of floats to exchange] (default: %d)\n", DEFAULT_DATA_SIZE);
	fprintf(stderr, "-S [time in millisecond of sleep between exchange, less than 1 second] (default: %d)\n", DEFAULT_SLEEP_TIME);
	fprintf(stderr, "-b : broadcasts instead of simple pair-wise ping-pongs (default: %s)\n", DEFAULT_METHOD ? "broadcast" : "ping pongs");
}

float *tab;
starpu_data_handle_t tab_handle;

int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;
	int i;

	int niter = DEFAULT_NITER;
	int data_size = DEFAULT_DATA_SIZE;
	int sleep_time = DEFAULT_SLEEP_TIME;
	int method = DEFAULT_METHOD;

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-n") == 0)
		{
			niter = atoi(argv[i+1]);
			if (niter <= 0)
			{
				fprintf(stderr, "%s: illegal argument %s\n", argv[0], argv[i]);
				usage();
				exit(0);
			}
			i++;
		}
		else if (strcmp(argv[i], "-s") == 0)
		{
			data_size = atoi(argv[i+1]);
			if (data_size <= 0)
			{
				fprintf(stderr, "%s: illegal argument %s\n", argv[0], argv[i]);
				usage();
				exit(0);
			}
			i++;
		}
		else if(strcmp(argv[i], "-S") == 0)
		{
			sleep_time = atoi(argv[i+1]);
			if (sleep_time <= 0 || sleep_time >= 1000)
			{
				fprintf(stderr, "%s: illegal argument %s\n", argv[0], argv[i]);
				usage();
				exit(0);
			}
			i++;
		}
		else if(strcmp(argv[i], "-b") == 0)
		{
			method = 1; // broadcasts
		}
		else
		{
			fprintf(stderr, "%s: illegal argument %s\n", argv[0], argv[i]);
			usage();
			exit(0);
		}
	}


	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need a even number of processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	if (rank == 0)
	{
		FPRINTF(stdout, "Number of iterations: %d\n", niter);
		FPRINTF(stdout, "Number of floats to exchange: %d\n", data_size);
		FPRINTF(stdout, "Sleep time between exchanges: %d milliseconds\n", sleep_time);
		if (method == 0)
			FPRINTF(stdout, "Method: ping pongs\n");
		else
			FPRINTF(stdout, "Method: broadcasts\n");
	}

	tab = calloc(data_size, sizeof(float));

	starpu_vector_data_register(&tab_handle, STARPU_MAIN_RAM, (uintptr_t)tab, data_size, sizeof(float));

	int loop;
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;
	int sender;
	int r;

	if (method == 0) // ping pongs
	{
		for (loop = 0; loop < niter; loop++)
		{
			if ((loop % 2) == (rank%2))
			{
				//FPRINTF_MPI(stderr, "Sending to %d\n", other_rank);
				ret = starpu_mpi_send(tab_handle, other_rank, loop, MPI_COMM_WORLD);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
			}
			else
			{
				MPI_Status status;
				//FPRINTF_MPI(stderr, "Receiving from %d\n", other_rank);
				ret = starpu_mpi_recv(tab_handle, other_rank, loop, MPI_COMM_WORLD, &status);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
			}

			starpu_sleep(sleep_time / 1000);
		}
	}
	else // broadcasts
	{
		for (loop = 0; loop < niter; loop++)
		{
			sender = loop % size;
			if (sender == rank)
			{
				for (r = 0; r < size; r++)
				{
					if (r != rank)
					{
						ret = starpu_mpi_send(tab_handle, r, (r * niter) + loop, MPI_COMM_WORLD);
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
						starpu_sleep(sleep_time / 1000);
					}
				}
			}
			else
			{
				MPI_Status status;
				ret = starpu_mpi_recv(tab_handle, sender, (rank * niter) + loop, MPI_COMM_WORLD, &status);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");

				for (r = 0; r < (size-1); r++)
					starpu_sleep(sleep_time / 1000);
			}
		}
	}

	starpu_data_unregister(tab_handle);
	free(tab);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
