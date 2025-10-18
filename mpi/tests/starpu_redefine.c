/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int main(int argc, char **argv)
{
#ifdef STARPU_NO_ASSERT
	// the code below will not fail as asserts are disabled
	// force the test to fail
	return 1;
#endif
	int ret;
	starpu_data_handle_t handle;
	int mpi_init;
	struct starpu_conf conf;

#ifdef STARPU_HAVE_VALGRIND_H
	if (RUNNING_ON_VALGRIND)
		return STARPU_TEST_SKIPPED;
#endif

	disable_coredump();

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	ret = starpu_mpi_init_conf(NULL, NULL, mpi_init, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&ret, 1, sizeof(int));
	starpu_mpi_datatype_register(handle, NULL, NULL);
	starpu_data_unregister(handle);

	starpu_mpi_shutdown();
enodev:
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
