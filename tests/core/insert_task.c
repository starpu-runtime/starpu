/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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

void func_cpu(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	float *x1 = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int ifactor;
	float ffactor;

	starpu_unpack_cl_args(_args, &ifactor, &ffactor);
        *x0 = *x0 * ifactor;
        *x1 = *x1 * ffactor;
}

starpu_codelet mycodelet = {
	.where = STARPU_CPU,
	.cpu_func = func_cpu,
        .nbuffers = 2
};

int main(int argc, char **argv)
{
        int x; float f;
        int i;
	int ifactor=15;
	float ffactor=25.0;
        starpu_data_handle data_handles[2];

	starpu_init(NULL);

	x = 10;
	starpu_variable_data_register(&data_handles[0], 0, (uintptr_t)&x, sizeof(x));
	f = 20.0;
	starpu_variable_data_register(&data_handles[1], 0, (uintptr_t)&f, sizeof(f));

        fprintf(stderr, "VALUES: %d %f\n", x, f);

        starpu_insert_task(&mycodelet,
			   STARPU_VALUE, &ifactor, sizeof(ifactor),
			   STARPU_VALUE, &ffactor, sizeof(ffactor),
                           STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
                           0);
        starpu_task_wait_for_all();

        for(i=0 ; i<2 ; i++) {
                starpu_data_acquire(data_handles[i], STARPU_R);
        }
        fprintf(stderr, "VALUES: %d %f\n", x, f);

	starpu_shutdown();

	return 0;
}
