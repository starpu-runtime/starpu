/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "helper.h"

/* This test shows how to use starpu_mpi_data_cpy() to copy data between two
 * handles, which are actually on different processes and require a MPI
 * communication.
 *
 * The test executes a ring: process 0 starts with a value 0, increments it,
 * send the value to process 1, process 1 incrementes the received value, it
 * sends it to process 2, and so on... To send the data to the following
 * process, we actually use starpu_data_cpy to send the data to *all*
 * processes.
 *
 * Warning: sending the same data to several processes triggers a broadcast
 * when NewMadeleine is used. It should be transparent, though.
 * */

#define DATA_TAG 666

void func_cpu(void *descr[], void *_args)
{
	int rank;
        int *value = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	starpu_codelet_unpack_args(_args, &rank);
	FPRINTF(stderr, "[rank %d] value in %d\n", rank, *value);
        (*value)++;
        FPRINTF(stderr, "[rank %d] value out %d\n", rank, *value);
}

struct starpu_codelet mycodelet =
{
        .cpu_funcs = {func_cpu},
        .nbuffers = 1,
        .modes = {STARPU_RW},
        .model = &starpu_perfmodel_nop,
	.name = "increment",
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
};

int main(int argc, char **argv)
{
        int size, rank;
        int ret;
        int value = 0;
        starpu_data_handle_t *data;
	struct starpu_conf conf;
        int mpi_init;
	int i;

        MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

        ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

        const int inc_count = size * 3; // Make the value be incremented by each process 3 times

        data = (starpu_data_handle_t*)malloc(size*sizeof(starpu_data_handle_t));
        for(i=0; i<size; i++)
	{
                if (i == rank)
                        starpu_variable_data_register(&data[i], STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(int));
                else
                        starpu_variable_data_register(&data[i], -1, (uintptr_t)NULL, sizeof(int));
                starpu_mpi_data_register_comm(data[i],  DATA_TAG + i,  i, MPI_COMM_WORLD);
        }

        for(i=0; i<inc_count; i++)
	{
		int j;

                ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data[i%size], STARPU_VALUE, &rank, sizeof(rank), 0);
		if (ret == -ENODEV) goto enodev_task;

                for(j = 0; j<size; j++)
		{
                        starpu_mpi_data_cpy(data[j], data[i%size], MPI_COMM_WORLD, 1, NULL, NULL);
                }
        }

        starpu_task_wait_for_all();

 enodev_task:
	for(i=0; i<size; i++)
	{
                starpu_data_unregister(data[i]);
	}
	free(data);

	if (ret == 0)
	{
		FPRINTF_MPI(stderr, "value after calculation: %d (expected %d)\n", value, inc_count);
		STARPU_ASSERT_MSG(value == inc_count, "[rank %d] value %d is not the expected value %d\n", rank, value, inc_count);
	}

        starpu_mpi_shutdown();

 enodev:
	if (!mpi_init)
		MPI_Finalize();

        return 0;
}
