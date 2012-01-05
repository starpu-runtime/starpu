/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012  Centre National de la Recherche Scientifique
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
#include "helper.h"

void func_cpu(void *descr[], __attribute__ ((unused)) void *_args)
{
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *y = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

        *x = *x + 1;
        *y = *y + 1;
}

struct starpu_codelet mycodelet_r_w =
{
	.where = STARPU_CPU,
	.cpu_funcs = {func_cpu, NULL},
        .nbuffers = 2
};

struct starpu_codelet mycodelet_rw_r =
{
	.where = STARPU_CPU,
	.cpu_funcs = {func_cpu, NULL},
        .nbuffers = 2
};

struct starpu_codelet mycodelet_rw_rw =
{
	.where = STARPU_CPU,
	.cpu_funcs = {func_cpu, NULL},
        .nbuffers = 2
};

struct starpu_codelet mycodelet_w_r =
{
	.where = STARPU_CPU,
	.cpu_funcs = {func_cpu, NULL},
        .nbuffers = 2
};

#define ACQUIRE_DATA \
        if (rank == 0) starpu_data_acquire(data_handlesx0, STARPU_R);    \
        if (rank == 1) starpu_data_acquire(data_handlesx1, STARPU_R);    \
        FPRINTF(stderr, "[%d] Values: %d %d\n", rank, x0, x1);

#define RELEASE_DATA \
        if (rank == 0) starpu_data_release(data_handlesx0); \
        if (rank == 1) starpu_data_release(data_handlesx1); \

#define CHECK_RESULT \
        if (rank == 0) assert(x0 == vx0[0] && x1 == vx1[0]); \
        if (rank == 1) assert(x0 == vx0[1] && x1 == vx1[1]);

int main(int argc, char **argv)
{
        int ret, rank, size, err;
        int x0=0, x1=0, vx0[2] = {x0, x0}, vx1[2]={x1,x1};
        starpu_data_handle_t data_handlesx0;
        starpu_data_handle_t data_handlesx1;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_initialize_extended(&rank, &size);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_initialize_extended");

        if (size != 2)
	{
		if (rank == 0) FPRINTF(stderr, "We need exactly 2 processes.\n");
                starpu_mpi_shutdown();
                starpu_shutdown();
                return STARPU_TEST_SKIPPED;
        }

        if (rank == 0)
	{
                starpu_variable_data_register(&data_handlesx0, 0, (uintptr_t)&x0, sizeof(x0));
                starpu_data_set_rank(data_handlesx0, rank);
		starpu_data_set_tag(data_handlesx0, 0);
                starpu_variable_data_register(&data_handlesx1, -1, (uintptr_t)NULL, sizeof(int));
                starpu_data_set_rank(data_handlesx1, 1);
		starpu_data_set_tag(data_handlesx1, 1);
        }
        else if (rank == 1)
	{
                starpu_variable_data_register(&data_handlesx1, 0, (uintptr_t)&x1, sizeof(x1));
                starpu_data_set_rank(data_handlesx1, rank);
		starpu_data_set_tag(data_handlesx1, 1);
                starpu_variable_data_register(&data_handlesx0, -1, (uintptr_t)NULL, sizeof(int));
                starpu_data_set_rank(data_handlesx0, 0);
		starpu_data_set_tag(data_handlesx0, 0);
        }

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet_r_w, STARPU_R, data_handlesx0, STARPU_W, data_handlesx1, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx1[1]++;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet_rw_r, STARPU_RW, data_handlesx0, STARPU_R, data_handlesx1, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx0[0] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet_rw_rw, STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1, 0);
        assert(err == -EINVAL);
        ACQUIRE_DATA;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet_rw_rw, STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1, STARPU_EXECUTE_ON_NODE, 1, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx0[0] ++ ; vx1[1] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet_rw_rw, STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1, STARPU_EXECUTE_ON_NODE, 0, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx0[0] ++ ; vx1[1] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        /* Here the value specified by the property STARPU_EXECUTE_ON_NODE is
           going to be ignored as the data model clearly specifies
           which task is going to execute the codelet */
        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet_r_w, STARPU_R, data_handlesx0, STARPU_W, data_handlesx1, STARPU_EXECUTE_ON_NODE, 12, 0);
        assert(err == 0);
        ACQUIRE_DATA;
        vx1[1] ++;
        CHECK_RESULT;
        RELEASE_DATA;

        /* Here the value specified by the property STARPU_EXECUTE_ON_NODE is
           going to be ignored as the data model clearly specifies
           which task is going to execute the codelet */
        err = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet_w_r, STARPU_W, data_handlesx0, STARPU_R, data_handlesx1, STARPU_EXECUTE_ON_NODE, 11, 0);
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

