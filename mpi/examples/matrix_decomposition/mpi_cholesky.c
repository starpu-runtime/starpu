/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

#include "mpi_cholesky.h"
#include "helper.h"

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	float ***bmat;
	int rank, nodes, ret;
	double timing, flops;
#ifndef STARPU_SIMGRID
	int correctness;
#endif

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &nodes);
	starpu_cublas_init();

	if (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() == 0)
	{
		if (rank == 0)
		{
			FPRINTF(stderr, "We need at least 1 CPU or CUDA worker.\n");
		}
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	parse_args(argc, argv, nodes);

	matrix_init(&bmat, rank, nodes, 1);
	matrix_display(bmat, rank, nodes);

	dw_cholesky(bmat, size/nblocks, rank, nodes, &timing, &flops);

#ifndef STARPU_SIMGRID
	matrix_display(bmat, rank, nodes);

	if (check && rank == 0)
		dw_cholesky_check_computation(bmat, rank, nodes, &correctness, &flops, 0.001);
#endif

	matrix_free(&bmat, rank, nodes, 1);

	starpu_cublas_shutdown();
	starpu_mpi_shutdown();

#ifndef STARPU_SIMGRID
	if (check && rank == 0)
		assert(correctness);
#endif

	if (rank == 0)
	{
		FPRINTF(stdout, "Computation time (in ms): %2.2f\n", timing/1000);
		FPRINTF(stdout, "Synthetic GFlops : %2.2f\n", (flops/timing/1000.0f));
	}

	return 0;
}
