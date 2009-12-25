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

#define N	1000

#define VECTORSIZE	1024

static pthread_mutex_t mutex;
static pthread_cond_t cond;

static unsigned finished = 0;

static unsigned cnt;

starpu_data_handle v_handle;
static unsigned *v;

static void callback(void *arg)
{
	unsigned res = STARPU_ATOMIC_ADD(&cnt, -1);

	//fprintf(stderr, "res ...%d\n", res);
	//fflush(stderr);

	if (res == 0)
	{
		pthread_mutex_lock(&mutex);
		finished = 1;
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
	}
}



static void codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
//	fprintf(stderr, "pif\n");
//	fflush(stderr);
}

static starpu_access_mode select_random_mode(void)
{
	int r = rand();

	switch (r % 3) {
		case 0:
			return STARPU_R;
		case 1:
			return STARPU_RW;
			//return STARPU_W;
		case 2:
			return STARPU_RW;
	};
}


static starpu_codelet cl = {
	.where = CORE|CUDA,
	.core_func = codelet_null,
	.cuda_func = codelet_null,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
	starpu_init(NULL);

	starpu_malloc_pinned_if_possible((void **)&v, VECTORSIZE*sizeof(unsigned));
	starpu_register_vector_data(&v_handle, 0, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned nworker = starpu_get_worker_count();

	cnt = nworker*N;

	unsigned iter, worker;
	for (iter = 0; iter < N; iter++)
	{
		for (worker = 0; worker < nworker; worker++)
		{
			/* synchronous prefetch */
			unsigned node = starpu_get_worker_memory_node(worker);
			starpu_prefetch_data_on_node(v_handle, node, 0);

			/* execute a task */
			struct starpu_task *task = starpu_task_create();
			task->cl = &cl;

			task->buffers[0].handle = v_handle;
			task->buffers[0].mode = select_random_mode();

			task->callback_func = callback;
			task->callback_arg = NULL;

			task->synchronous = 1;

			int ret = starpu_submit_task(task);
			if (ret == -ENODEV)
				goto enodev;
		}
	}

	pthread_mutex_lock(&mutex);
	if (!finished)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 0;
}
