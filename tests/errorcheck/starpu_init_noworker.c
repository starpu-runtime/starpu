/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Test that starpu_initialize returns ENODEV when no worker is available
 */

int main(int argc, char **argv)
{
	int ret;

	/* We try to initialize StarPU without any worker */
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.ncpus = 0;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
        conf.nmpi_ms = 0;

	/* starpu_init should return -ENODEV */
        ret = starpu_initialize(&conf, &argc, &argv);
        if (ret == -ENODEV)
                return EXIT_SUCCESS;
        else
        {
                unsigned ncpu = starpu_cpu_worker_get_count();
                unsigned ncuda = starpu_cuda_worker_get_count();
                unsigned nopencl = starpu_opencl_worker_get_count();
                unsigned nmic = starpu_mic_worker_get_count();
                unsigned nmpi_ms = starpu_mpi_ms_worker_get_count();
                FPRINTF(stderr, "StarPU has found :\n");
                FPRINTF(stderr, "\t%u CPU cores\n", ncpu);
                FPRINTF(stderr, "\t%u CUDA devices\n", ncuda);
                FPRINTF(stderr, "\t%u OpenCL devices\n", nopencl);
                FPRINTF(stderr, "\t%u MIC devices\n", nmic);
                FPRINTF(stderr, "\t%u MPI Master-Slaves devices\n", nmpi_ms);
                return EXIT_FAILURE;
        }


}
