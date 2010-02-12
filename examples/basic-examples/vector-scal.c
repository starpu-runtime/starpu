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

/*
 * This example demonstrates how to use StarPU to scale an array by a factor.
 * It shows how to manipulate data with StarPU's data management library.
 *  1- how to declare a piece of data to StarPU (starpu_register_vector_data)
 *  2- how to describe which data are accessed by a task (task->buffers[0])
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 */


#include <stdio.h>
#include <stdint.h>
#include <starpu.h>

#define	N	2048

/* This kernel takes a buffer and scales it by a constant factor */
static void scal_func(void *buffers[], void *arg)
{
	unsigned i;
	float *factor = arg;

#warning TODO update
	/* 
	 * The "buffers" array matches the task->buffers one: for instance
	 * task->buffers[0].handle is a handle that corresponds to a data with
	 * vector "interface". The starpu_data_interface_t is a union type with
	 * a field for each defined interface. Here, we manipulate the
	 * buffers[0].vector field: vector.nx gives the number of elements in
	 * the array, vector.ptr gives the location of the array (that was
	 * possibly migrated/replicated), and vector.elemsize gives the size of
	 * each elements.
	 */

	starpu_vector_interface_t *vector = buffers[0];

	/* length of the vector */
	unsigned n = vector->nx;

	/* get a pointer to the local copy of the vector : note that we have to
	 * cast it in (float *) since a vector could contain any type of
	 * elements so that the .ptr field is actually a uintptr_t */
	float *val = (float *)vector->ptr;

	/* scale the vector */
	for (i = 0; i < n; i++)
		val[i] *= *factor;
}

int main(int argc, char **argv)
{
	/* We consider a vector of float that is initialized just as any of C
 	 * data */
	float tab[N];
	unsigned i;
	for (i = 0; i < N; i++)
		tab[i] = 1.0f;

	fprintf(stderr, "BEFORE : First element was %f\n", tab[0]);

	/* Initialize StarPU with default configuration */
	starpu_init(NULL);

	/* Tell StaPU to associate the "tab" vector with the "tab_handle"
	 * identifier. When a task needs to access a piece of data, it should
	 * refer to the handle that is associated to it.
	 * In the case of the "vector" data interface:
	 *  - the first argument of the registration method is a pointer to the
	 *    handle that should describe the data
	 *  - the second argument is the memory node where the data (ie. "tab")
	 *    resides initially: 0 stands for an address in main memory, as
	 *    opposed to an adress on a GPU for instance.
	 *  - the third argument is the adress of the vector in RAM
	 *  - the fourth argument is the number of elements in the vector
	 *  - the fifth argument is the size of each element.
	 */
	starpu_data_handle tab_handle;
	starpu_register_vector_data(&tab_handle, 0, (uintptr_t)tab, N, sizeof(float));

	float factor = 3.14;

	/* create a synchronous task: any call to starpu_submit_task will block
 	 * until it is terminated */
	struct starpu_task *task = starpu_task_create();
	task->synchronous = 1;

	starpu_codelet cl = {
		.where = STARPU_CORE,
		/* CPU implementation of the codelet */
		.core_func = scal_func,
		.nbuffers = 1
	};

	task->cl = &cl;

	/* the codelet manipulates one buffer in RW mode */
	task->buffers[0].handle = tab_handle;
	task->buffers[0].mode = STARPU_RW;

	/* an argument is passed to the codelet, beware that this is a
	 * READ-ONLY buffer and that the codelet may be given a pointer to a
	 * COPY of the argument */
	task->cl_arg = &factor;
	task->cl_arg_size = sizeof(float);

	/* execute the task on any eligible computational ressource */
	starpu_submit_task(task);

	/* StarPU does not need to manipulate the array anymore so we can stop
 	 * monitoring it */
	starpu_delete_data(tab_handle);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	fprintf(stderr, "AFTER First element is %f\n", tab[0]);

	return 0;
}
