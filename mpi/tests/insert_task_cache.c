/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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

void func_cpu(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *x = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *y = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

        fprintf(stdout, "VALUES: %d %d\n", *x, *y);
        *x = (*x + *y) / 2;
}

starpu_codelet mycodelet = {
	.where = STARPU_CPU,
	.cpu_func = func_cpu,
        .nbuffers = 2
};

#define X     4
#define Y     5

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes) {
        return x % nb_nodes;
}


int main(int argc, char **argv)
{
        int rank, size, x, y;
        int value=0;
        unsigned matrix[X][Y];
        starpu_data_handle data_handles[X][Y];

	starpu_init(NULL);
	starpu_mpi_initialize_extended(&rank, &size);

        for(x = 0; x < X; x++) {
                for (y = 0; y < Y; y++) {
                        matrix[x][y] = (rank+1)*10 + value;
                        value++;
                }
        }
#if 0
        for(x = 0; x < X; x++) {
                fprintf(stdout, "[%d] ", rank);
                for (y = 0; y < Y; y++) {
                        fprintf(stdout, "%3d ", matrix[x][y]);
                }
                fprintf(stdout, "\n");
        }
#endif

        for(x = 0; x < X; x++) {
                for (y = 0; y < Y; y++) {
                        int mpi_rank = my_distrib(x, y, size);
                        if (mpi_rank == rank) {
                                //fprintf(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
                                starpu_variable_data_register(&data_handles[x][y], 0, (uintptr_t)&(matrix[x][y]), sizeof(unsigned));
                        }
                        else if (rank == mpi_rank+1 || rank == mpi_rank-1) {
                                /* I don't own that index, but will need it for my computations */
                                //fprintf(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
                                starpu_variable_data_register(&data_handles[x][y], -1, (uintptr_t)NULL, sizeof(unsigned));
                        }
                        else {
                                /* I know it's useless to allocate anything for this */
                                data_handles[x][y] = NULL;
                        }
                        if (data_handles[x][y])
                                starpu_data_set_rank(data_handles[x][y], mpi_rank);
                }
        }

        starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[1][1], STARPU_R, data_handles[0][1], 0);
        starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[3][1], STARPU_R, data_handles[0][1], 0);
        starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[0][1], STARPU_R, data_handles[0][0], 0);
        starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[3][1], STARPU_R, data_handles[0][1], 0);

        fprintf(stderr, "Waiting ...\n");
        starpu_task_wait_for_all();

        for(x = 0; x < X; x++) {
                for (y = 0; y < Y; y++) {
                        if (data_handles[x][y])
                                starpu_data_unregister(data_handles[x][y]);
                }
        }
	starpu_mpi_shutdown();
	starpu_shutdown();

#if 0
        for(x = 0; x < X; x++) {
                fprintf(stdout, "[%d] ", rank);
                for (y = 0; y < Y; y++) {
                        fprintf(stdout, "%3d ", matrix[x][y]);
                }
                fprintf(stdout, "\n");
        }
#endif

	return 0;
}
