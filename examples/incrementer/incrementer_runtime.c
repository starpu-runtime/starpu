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

#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>

/* for USE_CUDA */
#include <starpu_config.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

#include <starpu.h>

#define NITER	50000

extern void cuda_codelet_host(float *tab);

static starpu_data_handle my_float_state;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static float my_lovely_float[4] __attribute__ ((aligned (16))) = { 0.0f, 0.0f, 0.0f, 1664.0f}; 
static unsigned i;

void callback_func(void *argcb)
{
	unsigned cnt = STARPU_ATOMIC_ADD((unsigned *)argcb, 1);

	if (cnt == NITER) 
	{
		pthread_mutex_lock(&mutex);
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);

	}
}

void core_codelet(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].vector.ptr;

	val[0] += 1.0f; val[1] += 1.0f;
}

#ifdef USE_CUDA
void cuda_codelet(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].vector.ptr;

	cuda_codelet_host(val);
}

#endif

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned counter = 0;

	starpu_init(NULL);

	starpu_monitor_vector_data(&my_float_state, 0 /* home node */,
			(uintptr_t)&my_lovely_float, 4, sizeof(float));

	starpu_codelet cl =
	{
		/* CUBLAS stands for CUDA kernels controlled from the host */
		.where = CORE|CUBLAS,
		.core_func = core_codelet,
#ifdef USE_CUDA
		.cublas_func = &cuda_codelet,
#endif
		.nbuffers = 1
	};

	for (i = 0; i < NITER; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;
		
		task->callback_func = callback_func;
		task->callback_arg = &counter;

		task->buffers[0].state = my_float_state;
		task->buffers[0].mode = RW;

		starpu_submit_task(task);
	}

	pthread_mutex_lock(&mutex);
	pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	starpu_sync_data_with_mem(my_float_state);
	
	printf("array -> %f, %f, %f\n", my_lovely_float[0], 
			my_lovely_float[1], my_lovely_float[2]);
	
	if (my_lovely_float[0] != my_lovely_float[1] + my_lovely_float[2])
		return 1;
	
	starpu_shutdown();

	return 0;
}
