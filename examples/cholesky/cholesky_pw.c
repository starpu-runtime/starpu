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

#include "cholesky_tag.h"

// This version of cholesky uses parallel workers
// To enable them, one needs to set the environment variable STARPU_PARALLEL_WORKER_LEVEL
// the value must be a hwloc level optionally followed by : and a
// number to specify how many parallel workers should be created at
// the given level
// examples: STARPU_PARALLEL_WORKER_LEVEL=L3
//           STARPU_PARALLEL_WORKER_LEVEL=L3:2

struct starpu_parallel_worker_config *pw_config = NULL;

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

	char *env_pw_level = starpu_getenv("STARPU_PARALLEL_WORKER_LEVEL");
	if (env_pw_level == NULL)
	{
		FPRINTF(stderr, "Parallel workers are not enabled\n");
	}
	else
	{
		hwloc_obj_type_t pw_level;
		int pw_level_number = 1;
		char level[256];

		int n_argc = strchr(env_pw_level, ':') == NULL ? 1 : 2;
		int match = sscanf(env_pw_level, "%255[^:]:%d", level, &pw_level_number);

		if ((match != n_argc) || ((match == 2) && (pw_level_number < 0)))
		{
			fprintf(stderr, "error STARPU_PARALLEL_WORKER_LEVEL \"%s\"  does not match the format level[:number] where number > 0.\n", env_pw_level);
			exit(1);
		}

		if (hwloc_type_sscanf(level, &pw_level, NULL, 0) == -1)
		{
			fprintf(stderr, "error STARPU_PARALLEL_WORKER_LEVEL \"%s\"  does not match an hwloc level.\n", level);
			exit(1);
		}

		pw_config = starpu_parallel_worker_init(pw_level,
							STARPU_PARALLEL_WORKER_NB, pw_level_number,
							STARPU_PARALLEL_WORKER_POLICY_NAME, "dmda",
							0);

		if (pw_config == NULL)
		{
			fprintf(stderr, "error STARPU_PARALLEL_WORKER_LEVEL: cannot create a parallel worker at %s level.\n", level);
			exit(1);
		}

		starpu_parallel_worker_print(pw_config);
	}

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

	if (pw_config)
		starpu_parallel_worker_shutdown(pw_config);
	starpu_cusolver_shutdown();
	starpu_cublas_shutdown();
	starpu_shutdown();
}

int main(int argc, char **argv)
{
	return do_cholesky(argc, argv);
}
