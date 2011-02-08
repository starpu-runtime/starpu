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
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *x1 = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int *x2 = (int *)STARPU_VARIABLE_GET_PTR(descr[2]);
	int *y = (int *)STARPU_VARIABLE_GET_PTR(descr[3]);

//        fprintf(stderr, "-------> CODELET VALUES: %d %d %d %d\n", *x0, *x1, *x2, *y);
//
//        *x2 = 45;
//        *y = 144;
//
        fprintf(stderr, "-------> CODELET VALUES: %d %d %d %d\n", *x0, *x1, *x2, *y);
        *y = (*x0 + *x1) * -100;
        *x1 = 12;
        *x2 = 24;
        fprintf(stderr, "-------> CODELET VALUES: %d %d %d %d\n", *x0, *x1, *x2, *y);
}

starpu_codelet mycodelet = {
	.where = STARPU_CPU,
	.cpu_func = func_cpu,
        .nbuffers = 4
};

int main(int argc, char **argv)
{
        int rank, size, err;
        int x[3], y=0;
        int i;
        starpu_data_handle data_handles[4];

	starpu_init(NULL);
	starpu_mpi_initialize_extended(1, &rank, &size);

        if (rank == 0) {
                for(i=0 ; i<3 ; i++) {
                        x[i] = -10*(i+1);
                        starpu_variable_data_register(&data_handles[i], 0, (uintptr_t)&x[i], sizeof(x[i]));
                        starpu_data_set_rank(data_handles[i], rank);
                }
                y = -40;
                starpu_variable_data_register(&data_handles[3], -1, (uintptr_t)NULL, sizeof(int));
                starpu_data_set_rank(data_handles[3], 1);
        }
        else if (rank == 1) {
                for(i=0 ; i<3 ; i++) {
                        x[i] = -20;
                        starpu_variable_data_register(&data_handles[i], -1, (uintptr_t)NULL, sizeof(int));
                        starpu_data_set_rank(data_handles[i], 0);
                }
                y=-20;
                starpu_variable_data_register(&data_handles[3], 0, (uintptr_t)&y, sizeof(int));
                starpu_data_set_rank(data_handles[3], rank);
        }
        fprintf(stderr, "[%d] VALUES: %d %d %d %d\n", rank, x[0], x[1], x[2], y);

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet,
                                     STARPU_R, data_handles[0], STARPU_RW, data_handles[1],
                                     STARPU_W, data_handles[2],
                                     STARPU_W, data_handles[3],
                                     STARPU_EXECUTE, 1, 0);
        assert(err == 0);
        starpu_task_wait_for_all();

        if (rank == 0) {
                for(i=0 ; i<3 ; i++) {
                        starpu_data_acquire(data_handles[i], STARPU_R);
                }
        }
        else if (rank == 1) {
                starpu_data_acquire(data_handles[3], STARPU_R);
        }
        fprintf(stderr, "[%d] VALUES: %d %d %d %d\n", rank, x[0], x[1], x[2], y);

	starpu_mpi_shutdown();
	starpu_shutdown();

	return 0;
}

