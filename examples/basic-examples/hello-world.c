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
 * This examples demonstrates how to construct and submit a task to StarPU and
 * more precisely:
 *  - how to allocate a new task structure (starpu_task_create)
 *  - how to describe a multi-versionned computational kernel (ie. a codelet) 
 *  - how to pass an argument to the codelet (task->cl_arg)
 *  - how to declare a callback function that is called once the task has been
 *    executed
 *  - how to specify if starpu_submit_task is a blocking or non-blocking
 *    operation (task->synchronous)
 */

#include <stdio.h>
#include <stdint.h>
#include <starpu.h>

/* When the task is done, task->callback_func(task->callback_arg) is called. Any
 * callback function must have the prototype void (*)(void *).
 * NB: Callback are NOT allowed to perform potentially blocking operations */
void callback_func(void *callback_arg)
{
	printf("Callback function got argument %p\n", callback_arg);
}

/* Every implementation of a codelet must have this prototype, the first
 * argument (buffers) describes the buffers/streams that are managed by the
 * DSM; the second arguments references a read-only buffer that is passed as an
 * argument of the codelet (task->cl_arg). Here, "buffers" is unused as there
 * are no data input/output managed by the DSM (cl.nbuffers = 0) */
void cpu_func(starpu_data_interface_t *buffers, void *func_arg)
{
	float *array = func_arg;

	printf("Hello world (array = {%f, %f} )\n", array[0], array[1]);
}

starpu_codelet cl =
{
	/* this codelet may only be executed on a CPU, and its cpu
 	 * implementation is function "cpu_func" */
	.where = CORE,
	.core_func = cpu_func,
	/* the codelet does not manipulate any data that is managed
	 * by our DSM */
	.nbuffers = 0
};

int main(int argc, char **argv)
{
	/* initialize StarPU : passing a NULL argument means that we use
 	* default configuration for the scheduling policies and the number of
	* processors/accelerators */
	starpu_init(NULL);

	/* create a new task that is non-blocking by default : the task is not
	 * submitted to the scheduler until the starpu_submit_task function is
	 * called */
	struct starpu_task *task = starpu_task_create();

	/* the task uses codelet "cl" */
	task->cl = &cl;

	/* It is possible to use buffers that are not managed by the DSM to the
 	 * kernels: the second argument of the "cpu_func" function is a pointer to a
	 * buffer that contains information for the codelet (cl_arg stands for
	 * codelet argument). In the case of accelerators, it is possible that
	 * the codelet is given a pointer to a copy of that buffer: this buffer
	 * is read-only so that any modification is not passed to other copies
	 * of the buffer.  For this reason, a buffer passed as a codelet
	 * argument (cl_arg) is NOT a valid synchronization medium! */
	float array[2] = {1.0f, -1.0f};
	task->cl_arg = &array;
	task->cl_arg_size = 2*sizeof(float);
		
	/* once the task has been executed, callback_func(0x42)
	 * will be called on a CPU */
	task->callback_func = callback_func;
	task->callback_arg = (void*) (uintptr_t) 0x42;

	/* starpu_submit_task will be a blocking call */
	task->synchronous = 1;
	
	/* submit the task to StarPU */
	starpu_submit_task(task);
	
	/* terminate StarPU: statistics and other debug outputs are not
	 * guaranteed to be generated unless this function is called. Once it
	 * is called, it is not possible to submit tasks anymore, and the user
	 * is responsible for making sure all tasks have already been executed:
	 * calling starpu_shutdown() before the termination of all the tasks
	 * results in an undefined behaviour */
	starpu_shutdown();

	return 0;
}
