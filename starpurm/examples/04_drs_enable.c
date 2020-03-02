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

/* This example tests the proper initialization and shutdown of StarPURM. */

#include <stdio.h>
#include <starpurm.h>

int main(int argc, char *argv[])
{
	int drs_enabled;
	starpurm_initialize();
	drs_enabled = starpurm_drs_enabled_p();
	printf("drs enabled at startup: %d\n", drs_enabled);

	starpurm_set_drs_enable(NULL);
	drs_enabled = starpurm_drs_enabled_p();
	printf("drs state after explicit enable: %d\n", drs_enabled);

	starpurm_set_drs_disable(NULL);
	drs_enabled = starpurm_drs_enabled_p();
	printf("drs state after explicit disable: %d\n", drs_enabled);
	starpurm_shutdown();
	return 0;
}
