/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>

#include <starpu.h>

#ifdef STARPU_USE_GORDON
#include <gordon/null.h>
#endif

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)
#define TAG(i, j, iter)	((starpu_tag_t) ( ((uint64_t)(iter)<<48) |  ((uint64_t)(j)<<24) | (i)) )

starpu_codelet cl;

#define Ni	64
#define Nj	32
#define Nk	128

static unsigned ni = Ni, nj = Nj, nk = Nk;
static unsigned callback_cnt;
static unsigned iter = 0;

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

		if (strcmp(argv[i], "-j") == 0) {
		        char *argptr;
			nj = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-h") == 0) {
			printf("usage : %s [-iter iter] [-i i] [-j j]\n", argv[0]);
		}
	}
}

void callback_cpu(void *argcb);
static void express_deps(unsigned i, unsigned j, unsigned iter);

static void tag_cleanup_grid(unsigned ni, unsigned nj, unsigned iter)
{
	unsigned i,j;

	for (j = 0; j < nj; j++)
	for (i = 0; i < ni; i++)
	{
		starpu_tag_remove(TAG(i,j,iter));
	}


} 

static void create_task_grid(unsigned iter)
{
	unsigned i, j;

/*	FPRINTF(stderr, "start iter %d...\n", iter); */
	callback_cnt = (ni*nj);

	/* create non-entry tasks */
	for (j = 0; j < nj; j++)
	for (i = 1; i < ni; i++)
	{
		/* create a new task */
		struct starpu_task *task = starpu_task_create();
		task->callback_func = callback_cpu;
		/* jb->argcb = &coords[i][j]; */
		task->cl = &cl;
		task->cl_arg = NULL;

		task->use_tag = 1;
		task->tag_id = TAG(i, j, iter);

		/* express deps : (i,j) depends on (i-1, j-1) & (i-1, j+1) */
		express_deps(i, j, iter);

		starpu_task_submit(task);
	}

	/* create entry tasks */
	for (j = 0; j < nj; j++)
	{
		/* create a new task */
		struct starpu_task *task = starpu_task_create();
		task->callback_func = callback_cpu;
		task->cl = &cl;
		task->cl_arg = NULL;

		task->use_tag = 1;
		/* this is an entry task */
		task->tag_id = TAG(0, j, iter);

		starpu_task_submit(task);
	}

}


void callback_cpu(void *argcb __attribute__ ((unused)))
{
	unsigned newcnt = STARPU_ATOMIC_ADD(&callback_cnt, -1);	

	if (newcnt == 0)
	{
		if (++iter < nk)
		{
			/* cleanup old grids ... */
			if (iter > 2)
				tag_cleanup_grid(ni, nj, iter-2);

			/* create a new iteration */
			create_task_grid(iter);
		}
	}
}

void cpu_codelet(void *descr[] __attribute__((unused)),
			void *_args __attribute__ ((unused)))
{
/*	printf("execute task\n"); */
}

static void express_deps(unsigned i, unsigned j, unsigned iter)
{
	if (j > 0) {
		/* (i,j-1) exists */
		if (j < nj - 1)
		{
			/* (i,j+1) exists */
			starpu_tag_declare_deps(TAG(i,j,iter), 2, TAG(i-1,j-1,iter), TAG(i-1,j+1,iter));
		}
		else 
		{
			/* (i,j+1) does not exist */
			starpu_tag_declare_deps(TAG(i,j,iter), 1, TAG(i-1,j-1,iter));
		}
	}
	else {
		/* (i, (j-1) does not exist */
		if (j < nj - 1)
		{
			/* (i,j+1) exists */
			starpu_tag_declare_deps(TAG(i,j,iter), 1, TAG(i-1,j+1,iter));
		}
		else 
		{
			/* (i,j+1) does not exist */
			STARPU_ABORT();
		}
	}
}

int main(int argc __attribute__((unused)) , char **argv __attribute__((unused)))
{
	starpu_init(NULL);

#ifdef STARPU_USE_GORDON
	/* load an empty kernel and get its identifier */
	unsigned gordon_null_kernel = load_gordon_null_kernel();
#endif

	parse_args(argc, argv);

	FPRINTF(stderr, "ITER: %d\n", nk);

	cl.where = STARPU_CPU|STARPU_CUDA|STARPU_GORDON;
	cl.cpu_func = cpu_codelet;
	cl.cuda_func = cpu_codelet;
#ifdef STARPU_USE_GORDON
	cl.gordon_func = gordon_null_kernel;
#endif
	cl.nbuffers = 0;

	create_task_grid(0);

	starpu_task_wait_for_all();

	starpu_shutdown();

	FPRINTF(stderr, "TEST DONE ...\n");

	return 0;
}
