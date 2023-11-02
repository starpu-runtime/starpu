/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2013       Thibaut Lambert
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

/*
 * This version of the Cholesky factorization uses explicit dependency
 * declaration through dependency tags.
 * It also uses data partitioning to split the matrix into submatrices
 */

/* Note: this is using fortran ordering, i.e. column-major ordering, i.e.
 * elements with consecutive row number are consecutive in memory */

#include "cholesky_tag.h"

static int initialize_system(int argc, char **argv, float **A, unsigned pinned)
{
	int ret;
	int flags = STARPU_MALLOC_SIMULATION_FOLDED|STARPU_MALLOC_SIMULATION_UNIQUE;

#ifdef STARPU_HAVE_MAGMA
	magma_init();
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	init_sizes();

	parse_args(argc, argv);

	starpu_cublas_init();
	starpu_cusolver_init();

	if (pinned)
		flags |= STARPU_MALLOC_PINNED;
	starpu_malloc_flags((void **)A, (size_t)size_p*size_p*sizeof(float), flags);

	return 0;
}

static void shutdown_system(float **matA, unsigned dim, unsigned pinned)
{
	int flags = STARPU_MALLOC_SIMULATION_FOLDED|STARPU_MALLOC_SIMULATION_UNIQUE;
	if (pinned)
		flags |= STARPU_MALLOC_PINNED;

	starpu_free_flags(*matA, (size_t)dim*dim*sizeof(float), flags);

	starpu_cusolver_shutdown();
	starpu_cublas_shutdown();
	starpu_shutdown();
}

int main(int argc, char **argv)
{
	return do_cholesky(argc, argv);
}

