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
#include <stdint.h>
#include <starpu.h>

#define	N	2048

static void scal_func(starpu_data_interface_t *buffers, void *arg)
{
	unsigned i;
	float *factor = arg;

	/* length of the vector */
	unsigned n = buffers[0].vector.nx;

	/* get a pointer to the local copy of the vector */
	float *val = (float *)buffers[0].vector.ptr;

	for (i = 0; i < n; i++)
		val[i] *= *factor;
}

static starpu_codelet cl = {
	.where = CORE,
	.core_func = scal_func,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
	starpu_init(NULL);

	float tab[N];

	unsigned i;
	for (i = 0; i < N; i++)
		tab[i] = 1.0f;

	starpu_data_handle tab_handle;
	starpu_register_vector_data(&tab_handle, 0, (uintptr_t)tab, N, sizeof(float));

	float factor = 3.14;

	struct starpu_task *task = starpu_task_create();

	task->cl = &cl;

	task->buffers[0].handle = tab_handle;
	task->buffers[0].mode = STARPU_RW;

	task->cl_arg = &factor;
	task->cl_arg_size = sizeof(float);

	task->synchronous = 1;

	fprintf(stderr, "BEFORE : First element was %f\n", tab[0]);

	starpu_submit_task(task);

	fprintf(stderr, "AFTER First element is %f\n", tab[0]);

	starpu_shutdown();

	return 0;
}
