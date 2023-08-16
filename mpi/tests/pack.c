/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include "helper.h"

void func_dup_arg(void *descr[], void *_args)
{
	int *factor;
	char *c;
	int *x;
	size_t size;

	(void)descr;

	size_t psize = sizeof(int) + 3*sizeof(size_t) + sizeof(int) + sizeof(char) + 2*sizeof(int);
	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_unpack_arg_init(&state, _args, psize);

	starpu_codelet_dup_arg(&state, (void**)&factor, &size);
	STARPU_ASSERT_MSG(size == sizeof(*factor), "Expected size %ld != received size %ld\n", sizeof(*factor), size);

	starpu_codelet_dup_arg(&state, (void**)&c, &size);
	STARPU_ASSERT_MSG(size == sizeof(*c), "Expected size %ld != received size %ld\n", sizeof(*c), size);

	starpu_codelet_dup_arg(&state, (void**)&x, &size);
	STARPU_ASSERT_MSG(size == 2*sizeof(x[0]), "Expected size %ld != received size %ld\n", 2*sizeof(x[0]), size);

	starpu_codelet_unpack_arg_fini(&state);

	FPRINTF(stderr, "[codelet dup_arg] values: %d %c %d %d\n", *factor, *c, x[0], x[1]);
	assert(*factor == 12 && *c == 'n' && x[0] == 42 && x[1] == 24);
	free(factor);
	free(c);
	//free(x);
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_dup_arg},
	.cpu_funcs_name = {"func_dup_arg"},
        .nbuffers = 0
};

int main(int argc, char **argv)
{
	int ret;
	int rank, size;
	int mpi_init;
	int factor=12;
	int list[2]={42, 24};
	char c='n';

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_VALUE, &factor, sizeof(factor),
				     STARPU_VALUE, &c, sizeof(c),
				     STARPU_VALUE, list, 2*sizeof(list[0]),
				     0);
	if (ret != -ENODEV)
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
