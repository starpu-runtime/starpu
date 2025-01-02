/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void func_cpu(void *descr[], void *_args)
{
	(void) _args;
	(void) descr;

	FPRINTF_MPI(stderr, "Hello\n");
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &starpu_perfmodel_nop,
	.name = "insert_task_tags"
};

int main(int argc, char **argv)
{
	int ret, rank, err;
	int x=32;
	starpu_data_handle_t handle0;
	starpu_data_handle_t handle1;
	int64_t *value;
	struct starpu_conf conf;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	if (rank != 0 && rank != 1)
		goto end;

	starpu_variable_data_register(&handle0, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
	starpu_variable_data_register(&handle1, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));

	starpu_mpi_comm_get_attr(MPI_COMM_WORLD, STARPU_MPI_TAG_UB, &value, &err);
	assert(err == 1);

	starpu_mpi_data_register(handle0, (*value)-1, 1);
	starpu_mpi_data_register(handle1, (*value)-2, 1);

	err = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_EXECUTE_ON_NODE, 0,
				     STARPU_RW, handle0,
				     0);
	assert(err == 0);

	err = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_EXECUTE_ON_NODE, 1,
				     STARPU_RW, handle1,
				     0);
	assert(err == 0);

	FPRINTF_MPI(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();
	starpu_data_unregister(handle0);
	starpu_data_unregister(handle1);

end:
	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
