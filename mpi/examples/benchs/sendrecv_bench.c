/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Basic send receive benchmark.
 * Inspired a lot from NewMadeleine examples/benchmarks/nm_bench_sendrecv.c
 *
 * The option --bidir is available to do full-duplex communications.
 */

#include <starpu_mpi.h>
#include "helper.h"
#include "abstract_sendrecv_bench.h"

static inline void man()
{
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-h --help         display this help\n");
	fprintf(stderr, "\t-p                pause workers during benchmark\n");
	fprintf(stderr, "\t--bidir           full-duplex communications\n");
	fprintf(stderr, "\t--memnode-cuda    allocate message buffers on first CUDA device\n");
	fprintf(stderr, "\t--memnode-hip     allocate message buffers on first HIP device\n");
	exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
	int ret, rank, worldsize;
	int pause_workers = 0;
	int i;
	int bidir = 0;
	int mem_node = STARPU_MAIN_RAM;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-p") == 0)
		{
			pause_workers = 1;
			printf("Workers will be paused during benchmark.\n");
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			man();
		}
		else if (strcmp(argv[i], "--bidir") == 0)
		{
			bidir = 1;
			printf("Communications will be full-duplex.\n");
		}
		else if (strcmp(argv[i], "--memnode-cuda") == 0)
		{
			int worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
			if(worker_id == -1)
			{
				fprintf(stderr,"Error: asked for CUDA memory node allocation, but no cuda worker found.\n");
				starpu_mpi_shutdown();
				return STARPU_TEST_SKIPPED;
			}
			else
			{
				mem_node  = starpu_worker_get_memory_node(worker_id);
				fprintf(stderr,"Memory will be allocated on the first cuda worker.\n");
			}
		}
		else if (strcmp(argv[i], "--memnode-hip") == 0)
		{
			int worker_id = starpu_worker_get_by_type(STARPU_HIP_WORKER, 0);
			if(worker_id == -1)
			{
				fprintf(stderr,"Error: asked for HIP memory node allocation, but no hip worker found.\n");
				starpu_mpi_shutdown();
				return STARPU_TEST_SKIPPED;
			}
			else
			{
				mem_node  = starpu_worker_get_memory_node(worker_id);
				fprintf(stderr,"Memory will be allocated on the first hip worker.\n");
			}
		}
		else
		{
			fprintf(stderr,"Unrecognized option %s\n", argv[i]);
			man();
		}
	}

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	if (worldsize < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need 2 processes.\n");

		starpu_mpi_shutdown();

		return STARPU_TEST_SKIPPED;
	}

#if !defined(STARPU_LONG_CHECK)
	if (rank == 0)
	{
		printf("To have a more precise benchmark, configure StarPU with --enable-long-check\n");
	}
#endif

	if (pause_workers)
	{
		/* Pause workers for this bench: all workers polling for tasks has a strong impact on performances */
		starpu_pause();
	}

	ret = sendrecv_bench(rank, NULL, bidir, mem_node);
	if (ret == -ENODEV)
	{
		fprintf(stderr, "No device available\n");
	}

	if (pause_workers)
	{
		starpu_resume();
	}

	starpu_mpi_shutdown();

	return 0;
}
