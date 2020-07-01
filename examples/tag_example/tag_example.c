/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* This example shows how to use tags to define a grid of dependencies, shaped this way:
 *
 *           ...               ...
 *            v                 v
 * ... -> task (i,  j) --> task (i,  j+1) --> ...
 *            v                 v
 * ... -> task (i+1,j) --> task (i+1,j+1) --> ...
 *            v                 v
 *           ...               ...
 */

#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <signal.h>

#include <starpu.h>

#ifdef STARPU_HAVE_HELGRIND_H
#include <valgrind/helgrind.h>
#endif
#ifndef ANNOTATE_HAPPENS_BEFORE
#define ANNOTATE_HAPPENS_BEFORE(obj) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_AFTER
#define ANNOTATE_HAPPENS_AFTER(obj) ((void)0)
#endif

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define TAG(i, j, iter)	((starpu_tag_t) ( ((uint64_t)(iter)<<48) |  ((uint64_t)(j)<<24) | (i)) )

struct starpu_codelet cl;

#ifdef STARPU_QUICK_CHECK
#define Ni	32
#define Nj	32
#define Nk	32
#else
#define Ni	64
#define Nj	32
#define Nk	128
#endif

static unsigned ni = Ni, nj = Nj, nk = Nk;
static unsigned callback_cnt;
static unsigned iter = 0;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-iter") == 0)
		{
		        char *argptr;
			nk = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-i") == 0)
		{
		        char *argptr;
			ni = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-j") == 0)
		{
		        char *argptr;
			nj = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-h") == 0)
		{
			printf("usage : %s [-iter iter] [-i i] [-j j]\n", argv[0]);
		}
	}
}

void callback_cpu(void *argcb);
static void express_deps(unsigned i, unsigned j, unsigned iter);

static void tag_cleanup_grid(unsigned piter)
{
	unsigned i,j;

	for (j = 0; j < nj; j++)
	for (i = 0; i < ni; i++)
	{
		starpu_tag_remove(TAG(i,j,piter));
	}


}

static int create_task_grid(unsigned piter)
{
	unsigned i, j;
	int ret;

/*	FPRINTF(stderr, "start iter %d...\n", piter); */
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
		task->tag_id = TAG(i, j, piter);

		/* express deps : (i,j) depends on (i-1, j-1) & (i-1, j+1) */
		express_deps(i, j, piter);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) return 77;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
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
		task->tag_id = TAG(0, j, piter);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) return 77;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	return 0;
}


void callback_cpu(void *argcb)
{
	(void)argcb;
	unsigned newcnt = STARPU_ATOMIC_ADD(&callback_cnt, -1);
	ANNOTATE_HAPPENS_BEFORE(&callback_cnt);

	if (newcnt == 0)
	{
		ANNOTATE_HAPPENS_AFTER(&callback_cnt);
		if (++iter < nk)
		{
			/* cleanup old grids ... */
			if (iter > 2)
				tag_cleanup_grid(iter-2);

			/* create a new iteration */
			create_task_grid(iter);
		}
	}
}

void cpu_codelet(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
/*	printf("execute task\n"); */
}

static void express_deps(unsigned i, unsigned j, unsigned piter)
{
	if (j > 0)
	{
		/* (i,j-1) exists */
		if (j < nj - 1)
		{
			/* (i,j+1) exists */
			starpu_tag_declare_deps(TAG(i,j,piter), 2, TAG(i-1,j-1,piter), TAG(i-1,j+1,piter));
		}
		else
		{
			/* (i,j+1) does not exist */
			starpu_tag_declare_deps(TAG(i,j,piter), 1, TAG(i-1,j-1,piter));
		}
	}
	else
	{
		/* (i, (j-1) does not exist */
		if (j < nj - 1)
		{
			/* (i,j+1) exists */
			starpu_tag_declare_deps(TAG(i,j,piter), 1, TAG(i-1,j+1,piter));
		}
		else
		{
			/* (i,j+1) does not exist */
			STARPU_ABORT();
		}
	}
}

int main(int argc, char **argv)
{
	int ret;

#ifdef STARPU_HAVE_HELGRIND_H
	if (RUNNING_ON_VALGRIND) {
		ni /= 2;
		nj /= 2;
		nk /= 2;
	}
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	parse_args(argc, argv);

	FPRINTF(stderr, "ITER: %u\n", nk);

	starpu_codelet_init(&cl);
	cl.cpu_funcs[0] = cpu_codelet;
	cl.cpu_funcs_name[0] = "cpu_codelet";
	cl.cuda_funcs[0] = cpu_codelet;
	cl.opencl_funcs[0] = cpu_codelet;
	cl.nbuffers = 0;
	cl.name = "dummy";

	ret = create_task_grid(0);
	if (ret == 0)
	     starpu_task_wait_for_all();

	tag_cleanup_grid(nk-2);
	tag_cleanup_grid(nk-1);

	starpu_shutdown();

	FPRINTF(stderr, "TEST DONE ...\n");

	return ret;
}
