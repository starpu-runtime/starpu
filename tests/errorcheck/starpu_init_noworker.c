/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
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

int main(int argc, char **argv)
{
	int ret;	

	/* We try to initialize StarPU without any worker */
	struct starpu_conf conf = {
		.sched_policy_name = NULL, /* default */
		.ncpus = 0,
		.ncuda = 0,
                .nopencl = 0,
		.nspus = 0,
		.use_explicit_workers_bindid = 0,
		.use_explicit_workers_cuda_gpuid = 0,
		.use_explicit_workers_opencl_gpuid = 0,
		.calibrate = 0
	};

	/* starpu_init should return -ENODEV */
	ret = starpu_init(&conf);
	if (ret != -ENODEV)
		return -1;

	return 0;
}
