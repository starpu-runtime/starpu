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
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *y = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

        *x = *x + 1;
        *y = *y + 1;
}

starpu_codelet mycodelet = {
	.where = STARPU_CPU,
	.cpu_func = func_cpu,
        .nbuffers = 2
};

#define ACQUIRE_DATA \
        if (rank == 0) starpu_data_acquire(data_handlesx0, STARPU_R);    \
        if (rank == 1) starpu_data_acquire(data_handlesx1, STARPU_R);    \
        fprintf(stderr, "[%d] Values: %d %d\n", rank, x0, x1);

#define RELEASE_DATA \
        if (rank == 0) starpu_data_release(data_handlesx0); \
        if (rank == 1) starpu_data_release(data_handlesx1); \

#define CHECK_RESULT \
        if (rank == 0) assert(x0 == vx0[0] && x1 == vx1[0]); \
        if (rank == 1) assert(x0 == vx0[1] && x1 == vx1[1]);

int main(int argc, char **argv)
{
        int rank, size, err;
        int x0=0, x1=0, vx0[2] = {x0, x0}, vx1[2]={x1,x1};
        starpu_data_handle data_handlesx0;
        starpu_data_handle data_handlesx1;

	starpu_init(NULL);
	starpu_mpi_initialize_extended(1, &rank, &size);

        if (size != 2) {
		if (rank == 0) fprintf(stderr, "We need exactly 2 processes.\n");
                starpu_mpi_shutdown();
                starpu_shutdown();
                return 0;
        }

        if (rank == 0) {
                starpu_variable_data_register(&data_handlesx0, 0, (uintptr_t)&x0, sizeof(x0));
                starpu_data_set_rank(data_handlesx0, rank);
                starpu_variable_data_register(&data_handlesx1, -1, (uintptr_t)NULL, sizeof(int));
                starpu_data_set_rank(data_handlesx1, 1);
        }
        else if (rank == 1) {
                starpu_variable_data_register(&data_handlesx1, 0, (uintptr_t)&x1, sizeof(x1));
                starpu_data_set_rank(data_handlesx1, rank);
                starpu_variable_data_register(&data_handlesx0, -1, (uintptr_t)NULL, sizeof(int));
                starpu_data_set_rank(data_handlesx0, 0);
        }

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_R, data_handlesx0, STARPU_W, data_handlesx1, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx1[1]++;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handlesx0, STARPU_R, data_handlesx1, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx0[0] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1, 0);
        assert(err == -EINVAL);
        ACQUIRE_DATA;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1, STARPU_EXECUTE, 1, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx0[0] ++ ; vx1[1] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1, STARPU_EXECUTE, 0, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx0[0] ++ ; vx1[1] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        /* Here the value specified by the property STARPU_EXECUTE is
           going to be ignored as the data model clearly specifies
           which task is going to execute the codelet */
        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_R, data_handlesx0, STARPU_W, data_handlesx1, STARPU_EXECUTE, 12, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx1[1] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        /* Here the value specified by the property STARPU_EXECUTE is
           going to be ignored as the data model clearly specifies
           which task is going to execute the codelet */
        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_W, data_handlesx0, STARPU_R, data_handlesx1, STARPU_EXECUTE, 11, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx0[0] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        starpu_task_wait_for_all();
	starpu_mpi_shutdown();
	starpu_shutdown();

	return 0;
}

