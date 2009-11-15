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
#include <unistd.h>
#include <signal.h>

#include <starpu.h>

#ifdef USE_GORDON
#include <gordon/null.h>
#endif

#define Nrolls	4
#define SLEEP 1

#define TAG(i, iter)	((starpu_tag_t)  (((uint64_t)((iter)%Nrolls))<<32 | (i)) )

starpu_codelet cl;

#define Ni	64
#define Nk	256

static unsigned ni = Ni, nk = Nk;
static unsigned callback_cnt;
struct starpu_task **tasks[Nrolls];

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-iter") == 0) {
		        char *argptr;
			nk = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-i") == 0) {
		        char *argptr;
			ni = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-h") == 0) {
			printf("usage : %s [-iter iter] [-i i]\n", argv[0]);
		}
	}
}

void callback_core(void *argcb);

static void create_task_grid(unsigned iter)
{
	unsigned i;

	fprintf(stderr, "init iter %d ni %d...\n", iter, ni);

	callback_cnt = (ni);

	for (i = 0; i < ni; i++)
	{
		/* create a new task */
		struct starpu_task *task = tasks[iter][i] = starpu_task_create();

		task->cl = &cl;
		task->cl_arg = (void*)(uintptr_t) (i | (iter << 16));

		task->use_tag = 1;
		task->tag_id = TAG(i, iter);

		task->detach = 1;
		task->destroy = 0;

		if (i != 0)
			starpu_tag_declare_deps(TAG(i,iter), 1, TAG(i-1,iter));
	}

}

static void start_task_grid(unsigned iter)
{
	unsigned i;

	//fprintf(stderr, "start grid %d ni %d...\n", iter, ni);

	for (i = 0; i < ni; i++)
		starpu_submit_task(tasks[iter][i]);
}

void core_codelet(starpu_data_interface_t *descr, void *_args __attribute__((unused)))
{
	//int i = (uintptr_t) _args;
	//printf("doing %x\n", i);
	//usleep(SLEEP);
	//printf("done %x\n", i);
}

int main(int argc __attribute__((unused)) , char **argv __attribute__((unused)))
{
	unsigned i;

	starpu_init(NULL);

#ifdef USE_GORDON
	/* load an empty kernel and get its identifier */
	unsigned gordon_null_kernel = load_gordon_null_kernel();
#endif

	parse_args(argc, argv);

	cl.core_func = core_codelet;
	cl.cuda_func = core_codelet;
#ifdef USE_GORDON
	cl.gordon_func = gordon_null_kernel;
#endif
	cl.where = CORE|CUDA|GORDON;
	cl.nbuffers = 0;

	fprintf(stderr, "ITER : %d\n", nk);

	for (i = 0; i < Nrolls; i++) {
		tasks[i] = malloc(ni * sizeof(*tasks[i]));

		create_task_grid(i);
	}

	for (i = 0; i < nk; i++)
	{
		start_task_grid(i % Nrolls);

		if (i+1 >= Nrolls)
			/* Wait before re-using same tasks & tags */
			starpu_tag_wait(TAG(ni-1, i + 1));
	}

	starpu_shutdown();

	fprintf(stderr, "TEST DONE ...\n");

	return 0;
}
