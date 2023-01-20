/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#if !defined(STARPU_PARALLEL_WORKER)
int main(void)
{
	return 77;
}
#else

int main(void)
{
	int ret;
	struct starpu_cluster_machine *clusters;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* We regroup resources under each sockets into a parallel worker. We express a partition
	 * of one socket to create two internal parallel workers */
	clusters = starpu_cluster_machine(HWLOC_OBJ_SOCKET,
					  STARPU_CLUSTER_POLICY_NAME, "dmdas",
					  STARPU_PARALLEL_WORKER_PARTITION_ONE,
					  STARPU_PARALLEL_WORKER_NEW,
					  STARPU_PARALLEL_WORKER_NB, 2,
					  STARPU_PARALLEL_WORKER_NCORES, 1,
					  0);
	if (clusters != NULL)
	{
		starpu_cluster_print(clusters);
		starpu_uncluster_machine(clusters);
	}

	starpu_shutdown();
	return 0;
}
#endif
