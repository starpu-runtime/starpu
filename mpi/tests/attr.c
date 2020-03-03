/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <inttypes.h>

int main(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	int flag;
	int64_t *value;
	int64_t rvalue;

	starpu_mpi_comm_get_attr(MPI_COMM_WORLD, 42, NULL, &flag);
	STARPU_ASSERT_MSG(flag == 0, "starpu_mpi_comm_get_attr was called with invalid argument\n");

	starpu_mpi_comm_get_attr(MPI_COMM_WORLD, STARPU_MPI_TAG_UB, &value, &flag);
	STARPU_ASSERT_MSG(flag == 1, "starpu_mpi_comm_get_attr was called with valid argument\n");

	rvalue = *value;
	FPRINTF(stderr, "Value: %"PRIi64"\n", *value);
	FPRINTF(stderr, "Value: %"PRIi64"\n", rvalue);

	return 0;
}
