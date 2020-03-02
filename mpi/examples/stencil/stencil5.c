/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <math.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define FPRINTF_MPI(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) { \
    						int _disp_rank; starpu_mpi_comm_rank(MPI_COMM_WORLD, &_disp_rank);       \
                                                fprintf(ofile, "[%d][starpu_mpi][%s] " fmt , _disp_rank, __starpu_func__ ,## __VA_ARGS__); \
                                                fflush(ofile); }} while(0);

void stencil5_cpu(void *descr[], void *_args)
{
	(void)_args;
	float *xy = (float *)STARPU_VARIABLE_GET_PTR(descr[0]);
	float *xm1y = (float *)STARPU_VARIABLE_GET_PTR(descr[1]);
	float *xp1y = (float *)STARPU_VARIABLE_GET_PTR(descr[2]);
	float *xym1 = (float *)STARPU_VARIABLE_GET_PTR(descr[3]);
	float *xyp1 = (float *)STARPU_VARIABLE_GET_PTR(descr[4]);

//	fprintf(stdout, "VALUES: %2.2f %2.2f %2.2f %2.2f %2.2f\n", *xy, *xm1y, *xp1y, *xym1, *xyp1);
	*xy = (*xy + *xm1y + *xp1y + *xym1 + *xyp1) / 5;
//	fprintf(stdout, "VALUES: %2.2f %2.2f %2.2f %2.2f %2.2f\n", *xy, *xm1y, *xp1y, *xym1, *xyp1);
}

struct starpu_codelet stencil5_cl =
{
	.cpu_funcs = {stencil5_cpu},
	.nbuffers = 5,
	.modes = {STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R},
	.model = &starpu_perfmodel_nop,
};

#ifdef STARPU_QUICK_CHECK
#  define NITER_DEF	5
#  define X         	4
#  define Y         	4
#elif !defined(STARPU_LONG_CHECK)
#  define NITER_DEF	10
#  define X         	5
#  define Y         	5
#else
#  define NITER_DEF	100
#  define X         	20
#  define Y         	20
#endif

int display = 0;
int niter = NITER_DEF;

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes)
{
	/* Block distrib */
	return ((int)(x / sqrt(nb_nodes) + (y / sqrt(nb_nodes)) * sqrt(nb_nodes))) % nb_nodes;
}

/* Shifted distribution, for migration example */
int my_distrib2(int x, int y, int nb_nodes)
{
	return (my_distrib(x, y, nb_nodes) + 1) % nb_nodes;
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-iter") == 0)
		{
			char *argptr;
			niter = strtol(argv[++i], &argptr, 10);
		}
		if (strcmp(argv[i], "-display") == 0)
		{
			display = 1;
		}
	}
}

int main(int argc, char **argv)
{
	int my_rank, size, x, y, loop;
	float mean=0;
	float matrix[X][Y];
	starpu_data_handle_t data_handles[X][Y];
	int ret;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return 77;
	}

	parse_args(argc, argv);

	/* Initial data values */
	starpu_srand48((long int)time(NULL));
	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			matrix[x][y] = (float)starpu_drand48();
			mean += matrix[x][y];
		}
	}
	mean /= (X*Y);

	if (display)
	{
		FPRINTF_MPI(stdout, "mean=%2.2f\n", mean);
		for(x = 0; x < X; x++)
		{
			fprintf(stdout, "[%d] ", my_rank);
			for (y = 0; y < Y; y++)
			{
				fprintf(stdout, "%2.2f ", matrix[x][y]);
			}
			fprintf(stdout, "\n");
		}
	}

	/* Initial distribution */
	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			int mpi_rank = my_distrib(x, y, size);
			if (mpi_rank == my_rank)
			{
				//FPRINTF(stderr, "[%d] Owning data[%d][%d]\n", my_rank, x, y);
				starpu_variable_data_register(&data_handles[x][y], 0, (uintptr_t)&(matrix[x][y]), sizeof(float));
			}
			else if (my_rank == my_distrib(x+1, y, size) || my_rank == my_distrib(x-1, y, size)
				 || my_rank == my_distrib(x, y+1, size) || my_rank == my_distrib(x, y-1, size))
			{
				/* I don't own this index, but will need it for my computations */
				//FPRINTF(stderr, "[%d] Neighbour of data[%d][%d]\n", my_rank, x, y);
				starpu_variable_data_register(&data_handles[x][y], -1, (uintptr_t)NULL, sizeof(float));
			}
			else
			{
				/* I know it's useless to allocate anything for this */
				data_handles[x][y] = NULL;
			}
			if (data_handles[x][y])
			{
				starpu_data_set_coordinates(data_handles[x][y], 2, x, y);
				starpu_mpi_data_register(data_handles[x][y], (y*X)+x, mpi_rank);
			}
		}
	}

	/* First computation with initial distribution */
	for(loop=0 ; loop<niter; loop++)
	{
		starpu_iteration_push(loop);

		for (x = 1; x < X-1; x++)
		{
			for (y = 1; y < Y-1; y++)
			{
				starpu_mpi_task_insert(MPI_COMM_WORLD, &stencil5_cl, STARPU_RW, data_handles[x][y],
						       STARPU_R, data_handles[x-1][y], STARPU_R, data_handles[x+1][y],
						       STARPU_R, data_handles[x][y-1], STARPU_R, data_handles[x][y+1],
						       0);
			}
		}
		starpu_iteration_pop();
	}
	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	/* Now migrate data to a new distribution */

	/* First register newly needed data */
	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			int mpi_rank = my_distrib2(x, y, size);
			if (!data_handles[x][y] && (mpi_rank == my_rank
				 || my_rank == my_distrib2(x+1, y, size) || my_rank == my_distrib2(x-1, y, size)
				 || my_rank == my_distrib2(x, y+1, size) || my_rank == my_distrib2(x, y-1, size)))
			{
				/* Register newly-needed data */
				starpu_variable_data_register(&data_handles[x][y], -1, (uintptr_t)NULL, sizeof(float));
				starpu_mpi_data_register(data_handles[x][y], (y*X)+x, mpi_rank);
			}
			if (data_handles[x][y] && mpi_rank != starpu_mpi_data_get_rank(data_handles[x][y]))
				/* Migrate the data */
				starpu_mpi_data_migrate(MPI_COMM_WORLD, data_handles[x][y], mpi_rank);
		}
	}

	/* Second computation with new distribution */
	for(loop=0 ; loop<niter; loop++)
	{
		starpu_iteration_push(niter + loop);

		for (x = 1; x < X-1; x++)
		{
			for (y = 1; y < Y-1; y++)
			{
				starpu_mpi_task_insert(MPI_COMM_WORLD, &stencil5_cl, STARPU_RW, data_handles[x][y],
						       STARPU_R, data_handles[x-1][y], STARPU_R, data_handles[x+1][y],
						       STARPU_R, data_handles[x][y-1], STARPU_R, data_handles[x][y+1],
						       0);
			}
		}
		starpu_iteration_pop();
	}
	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	/* Unregister data */
	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			if (data_handles[x][y])
			{
				int mpi_rank = my_distrib(x, y, size);
				/* Get back data to original place where the user-provided buffer is. */
				starpu_mpi_data_migrate(MPI_COMM_WORLD, data_handles[x][y], mpi_rank);
				/* And unregister it */
				starpu_data_unregister(data_handles[x][y]);
			}
		}
	}

	starpu_mpi_shutdown();

	if (display)
	{
		FPRINTF(stdout, "[%d] mean=%2.2f\n", my_rank, mean);
		for(x = 0; x < X; x++)
		{
			FPRINTF(stdout, "[%d] ", my_rank);
			for (y = 0; y < Y; y++)
			{
				FPRINTF(stdout, "%2.2f ", matrix[x][y]);
			}
			FPRINTF(stdout, "\n");
		}
	}

	return 0;
}
