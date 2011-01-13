/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu_mpi.h>
#include <math.h>

void stencil5_cpu(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *xy = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *xm1y = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	unsigned *xp1y = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[2]);
	unsigned *xym1 = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[3]);
	unsigned *xyp1 = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[4]);

        fprintf(stdout, "VALUES: %d %d %d %d %d\n", *xy, *xm1y, *xp1y, *xym1, *xyp1);
        *xy += *xm1y + *xp1y + *xym1 + *xyp1;
}

starpu_codelet stencil5_cl = {
	.where = STARPU_CPU,
	.cpu_func = stencil5_cpu,
        .nbuffers = 5
};

#define X 3
#define Y 5

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes) {
	/* Cyclic distrib */
	return ((int)(x / sqrt(nb_nodes) + (y / sqrt(nb_nodes)) * sqrt(nb_nodes))) % nb_nodes;
        //	/* Linear distrib */
        //	return x / sqrt(nb_nodes) + (y / sqrt(nb_nodes)) * X;
}


int main(int argc, char **argv)
{
        int rank, size, x, y;
        int value=10;
        unsigned matrix[X][Y];
        starpu_data_handle data_handles[X][Y];

	starpu_init(NULL);
	starpu_mpi_initialize_extended(1, &rank, &size);

        for(x = 0; x < X; x++) {
                for (y = 0; y < Y; y++) {
                        matrix[x][y] = value;
                        value++;
                }
        }
        for(x = 0; x < X; x++) {
                for (y = 0; y < Y; y++) {
                        fprintf(stdout, "%4d ", matrix[x][y]);
                }
                fprintf(stdout, "\n");
        }

        for(x = 0; x < X; x++) {
                for (y = 0; y < Y; y++) {
                        int mpi_rank = my_distrib(x, y, size);
                        if (mpi_rank == rank) {
                                fprintf(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
                                starpu_variable_data_register(&data_handles[x][y], 0, (uintptr_t)&(matrix[x][y]), sizeof(unsigned));
                        }
                        else if (rank == mpi_rank+1 || rank == mpi_rank-1) {
                                /* I don't own that index, but will need it for my computations */
                                //fprintf(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
#warning should use NULL instead of matrix[x][y]
                                starpu_variable_data_register(&data_handles[x][y], 0, (uintptr_t)&(matrix[x][y]), sizeof(unsigned));
                        }
                        else {
                                /* I know it's useless to allocate anything for this */
                                data_handles[x][y] = NULL;
                        }
                        if (data_handles[x][y])
                                starpu_data_set_rank(data_handles[x][y], mpi_rank);
                }
        }

	for (x = 1; x < X-1; x++) {
                for (y = 1; y < Y-1; y++) {
                        starpu_mpi_insert_task(MPI_COMM_WORLD, &stencil5_cl, STARPU_RW, data_handles[x][y],
                                               STARPU_R, data_handles[x-1][y], STARPU_R, data_handles[x+1][y],
                                               STARPU_R, data_handles[x][y-1], STARPU_R, data_handles[x][y+1],
                                               0);
                }
        }

	starpu_mpi_shutdown();
	starpu_shutdown();

        for(x = 0; x < X; x++) {
                for (y = 0; y < Y; y++) {
                        fprintf(stdout, "%4d ", matrix[x][y]);
                }
                fprintf(stdout, "\n");
        }

	return 0;
}
