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

#include <semaphore.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>

/* for USE_CUDA */
#include <starpu_config.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cublas.h>
#endif

#include <starpu.h>

#define NITER	50000

data_handle my_float_state;
data_handle unity_state;

sem_t sem;

unsigned size __attribute__ ((aligned (16))) = 4*sizeof(float);

float my_lovely_float[4] __attribute__ ((aligned (16))) = { 0.0f, 0.0f, 0.0f, 1664.0f}; 
float unity[4] __attribute__ ((aligned (16))) = { 1.0f, 0.0f, 1.0f, 0.0f };
unsigned i;

void callback_func(void *argcb)
{
	unsigned cnt = STARPU_ATOMIC_ADD((unsigned *)argcb, 1);

	if (cnt == NITER) 
	{
		sem_post(&sem);
	}
}

void core_codelet(data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].vector.ptr;

	val[0] += 1.0f; val[1] += 1.0f;
}

#ifdef USE_CUDA
void cublas_codelet(data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	float *val = (float *)buffers[0].vector.ptr;
	float *dunity = (float *)buffers[1].vector.ptr;

	cublasSaxpy(3, 1.0f, dunity, 1, val, 1);
}
#endif

#ifdef USE_CUDA
static struct cuda_module_s cuda_module;
static struct cuda_function_s cuda_function;

static cuda_codelet_t cuda_codelet;

void initialize_cuda(void)
{
	char module_path[1024];
	sprintf(module_path, 
		"%s/examples/cuda/incrementer_cuda.cubin", STARPUDIR);
	char *function_symbol = "cuda_incrementer";

	starpu_init_cuda_module(&cuda_module, module_path);
	starpu_init_cuda_function(&cuda_function, &cuda_module, function_symbol);

	cuda_codelet.func = &cuda_function;

	cuda_codelet.gridx = 1;
	cuda_codelet.gridy = 1;

	cuda_codelet.blockx = 1;
	cuda_codelet.blocky = 1;

	cuda_codelet.shmemsize = 1024;
}
#endif

void init_data(void)
{
	monitor_vector_data(&my_float_state, 0 /* home node */, (uintptr_t)&my_lovely_float, 4, sizeof(float));

	monitor_vector_data(&unity_state, 0 /* home node */, (uintptr_t)&unity, 4, sizeof(float));
}

int main(__attribute__ ((unused)) int argc, __attribute__ ((unused)) char **argv)
{
	unsigned counter = 0;

	starpu_init();
	fprintf(stderr, "StarPU initialized ...\n");

	sem_init(&sem, 0, 0);

	init_data();

#ifdef USE_CUDA
	initialize_cuda();
#endif

	starpu_codelet cl =
	{
		.core_func = core_codelet,
		.where = CORE|CUDA|GORDON,
#ifdef USE_CUDA
		.cuda_func = &cuda_codelet,
#endif
#ifdef USE_GORDON
		.gordon_func = SPU_FUNC_ADD,
#endif
		.nbuffers = 2
	};

	for (i = 0; i < NITER; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;
		
		task->callback_func = callback_func;
		task->callback_arg = &counter;

		task->cl_arg = &size;
		task->cl_arg_size = sizeof(unsigned);

		task->buffers[0].state = my_float_state;
		task->buffers[0].mode = RW;
		task->buffers[1].state = unity_state; 
		task->buffers[1].mode = R;

		task->use_tag = 0;
		task->synchronous = 0;

		starpu_submit_task(task);
	}

	sem_wait(&sem);

//	/* stop monitoring data and grab it in RAM */
//	unpartition_data(&my_float_state, 0);
//
//	delete_data(&my_float_state);

	sync_data_with_mem(my_float_state);
	
	printf("array -> %f, %f, %f\n", my_lovely_float[0], 
			my_lovely_float[1], my_lovely_float[2]);
//	printf("stopping ... cnt was %d i %d\n", cnt, i);
	
	if (my_lovely_float[0] != my_lovely_float[1] + my_lovely_float[2])
		return 1;
	
	starpu_shutdown();

	return 0;
}
