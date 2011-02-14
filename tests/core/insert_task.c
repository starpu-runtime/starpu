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

void func_cpu(void *descr[], __attribute__ ((unused)) void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *x1 = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

        *x0 = *x0 * 10;
        *x1 = *x1 * 10;
}

starpu_codelet mycodelet = {
	.where = STARPU_CPU,
	.cpu_func = func_cpu,
        .nbuffers = 2
};

int main(int argc, char **argv)
{
        int x[2];
        int i;
        starpu_data_handle data_handles[2];

	starpu_init(NULL);

        for(i=0 ; i<2 ; i++) {
                x[i] = 10*(i+1);
                starpu_variable_data_register(&data_handles[i], 0, (uintptr_t)&x[i], sizeof(x[i]));
        }

        starpu_insert_task(&mycodelet,
                           STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
                           0);
        starpu_task_wait_for_all();

        for(i=0 ; i<2 ; i++) {
                starpu_data_acquire(data_handles[i], STARPU_R);
        }
        fprintf(stderr, "VALUES: %d %d\n", x[0], x[1]);

	starpu_shutdown();

	return 0;
}
