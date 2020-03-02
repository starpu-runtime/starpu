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
#include <starpu_mpi_lb.h>
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

void get_neighbors(int **neighbor_ids, int *nneighbors)
{
	int rank, size;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size <= 2)
	{
		*nneighbors = 1;
		*neighbor_ids = malloc(sizeof(int));
		*neighbor_ids[0] = rank==size-1?0:rank+1;
		fprintf(stderr, "rank %d has neighbor %d\n", rank, *neighbor_ids[0]);
	}
	else
	{
		*nneighbors = 2;
		*neighbor_ids = malloc(2*sizeof(int));
		(*neighbor_ids)[0] = rank==size-1?0:rank+1;
		(*neighbor_ids)[1] = rank==0?size-1:rank-1;
		fprintf(stderr, "rank %d has neighbor %d and %d\n", rank, (*neighbor_ids)[0], (*neighbor_ids)[1]);
	}
}

struct data_node
{
	starpu_data_handle_t data_handle;
	int node;
};

struct data_node data_nodes[X][Y];

void get_data_unit_to_migrate(starpu_data_handle_t **handle_unit, int *nhandles, int dst_node)
{
	int rank, x, y;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	fprintf(stderr, "Looking to move data from %d to %d\n", rank, dst_node);
	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			if (data_nodes[x][y].node == rank)
			{
				*handle_unit = malloc(sizeof(starpu_data_handle_t));
				*handle_unit[0] = data_nodes[x][y].data_handle;
				*nhandles = 1;
				data_nodes[x][y].node = dst_node;
				return;
			}
		}
	}
	*nhandles = 0;
}

int main(int argc, char **argv)
{
	int my_rank, size, x, y, loop;
	float mean=0;
	float matrix[X][Y];
	struct starpu_mpi_lb_conf itf;
	int ret;

	itf.get_neighbors = get_neighbors;
	itf.get_data_unit_to_migrate = get_data_unit_to_migrate;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size > 2)
	{
		FPRINTF(stderr, "Only works with 2 nodes\n");
		starpu_mpi_shutdown();
		return 77;
	}
	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return 77;
	}

	{
		char sleep_thr[10];
		snprintf(sleep_thr, 10, "%d", Y);
		setenv("LB_HEAT_SLEEP_THRESHOLD", sleep_thr, 1);
	}
	starpu_mpi_lb_init("heat", &itf);

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
			data_nodes[x][y].node = my_distrib(x, y, size);
			if (data_nodes[x][y].node == my_rank)
			{
				//FPRINTF(stderr, "[%d] Owning data[%d][%d]\n", my_rank, x, y);
				starpu_variable_data_register(&data_nodes[x][y].data_handle, 0, (uintptr_t)&(matrix[x][y]), sizeof(float));
			}
			else if (my_rank == my_distrib(x+1, y, size) || my_rank == my_distrib(x-1, y, size)
				 || my_rank == my_distrib(x, y+1, size) || my_rank == my_distrib(x, y-1, size))
			{
				/* I don't own this index, but will need it for my computations */
				//FPRINTF(stderr, "[%d] Neighbour of data[%d][%d]\n", my_rank, x, y);
				starpu_variable_data_register(&data_nodes[x][y].data_handle, -1, (uintptr_t)NULL, sizeof(float));
			}
			else
			{
				/* I know it's useless to allocate anything for this */
				data_nodes[x][y].data_handle = NULL;
			}
			if (data_nodes[x][y].data_handle)
			{
				starpu_data_set_coordinates(data_nodes[x][y].data_handle, 2, x, y);
				starpu_mpi_data_register(data_nodes[x][y].data_handle, (y*X)+x, data_nodes[x][y].node);
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
				starpu_mpi_task_insert(MPI_COMM_WORLD, &stencil5_cl, STARPU_RW, data_nodes[x][y].data_handle,
						       STARPU_R, data_nodes[x-1][y].data_handle, STARPU_R, data_nodes[x+1][y].data_handle,
						       STARPU_R, data_nodes[x][y-1].data_handle, STARPU_R, data_nodes[x][y+1].data_handle,
						       STARPU_TAG_ONLY, ((starpu_tag_t)Y)*x + y,
						       0);
			}
		}
		starpu_iteration_pop();
	}
	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	// The load balancer needs to be shutdown before unregistering data as it needs access to them
	starpu_mpi_lb_shutdown();

	/* Unregister data */
	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			if (data_nodes[x][y].data_handle)
			{
				starpu_data_unregister(data_nodes[x][y].data_handle);
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
