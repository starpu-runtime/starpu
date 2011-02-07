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

int main(int argc, char **argv)
{
        int rank, size;
        unsigned x0, x1;
        starpu_data_handle data_handlesx0;
        starpu_data_handle data_handlesx1;

	starpu_init(NULL);
	starpu_mpi_initialize_extended(1, &rank, &size);

        if (rank == 0) {
                starpu_variable_data_register(&data_handlesx0, 0, (uintptr_t)&x0, sizeof(x0));
                starpu_data_set_rank(data_handlesx0, rank);
                starpu_variable_data_register(&data_handlesx1, -1, (uintptr_t)NULL, sizeof(unsigned));
                starpu_data_set_rank(data_handlesx1, 1);
        }
        else if (rank == 1) {
                starpu_variable_data_register(&data_handlesx1, 1, (uintptr_t)&x1, sizeof(x1));
                starpu_data_set_rank(data_handlesx1, rank);
                starpu_variable_data_register(&data_handlesx0, -1, (uintptr_t)NULL, sizeof(unsigned));
                starpu_data_set_rank(data_handlesx0, 0);
        }

        int err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1);
        if (err == -EINVAL) {
                fprintf(stderr, "starpu_mpi_insert_task failed as expected\n");
        }
        else {
                return 1;
        }

        starpu_task_wait_for_all();
	starpu_mpi_shutdown();
	starpu_shutdown();

	return 0;
}

