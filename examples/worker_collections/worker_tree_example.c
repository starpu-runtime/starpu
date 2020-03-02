/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This shows how to manipulate worker trees.
 */

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#if !defined(STARPU_HAVE_HWLOC)
#warning hwloc is not enabled. Skipping test
int main(int argc, char **argv)
{
	return 77;
}
#else

int main()
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	int procs[STARPU_NMAXWORKERS];
	unsigned ncpus =  starpu_cpu_worker_get_count();
        starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs, ncpus);

	struct starpu_worker_collection *co = (struct starpu_worker_collection*)calloc(1, sizeof(struct starpu_worker_collection));
	co->has_next = worker_tree.has_next;
	co->get_next = worker_tree.get_next;
	co->add = worker_tree.add;
	co->remove = worker_tree.remove;
	co->init = worker_tree.init;
	co->deinit = worker_tree.deinit;
	co->init_iterator = worker_tree.init_iterator;
	co->type = STARPU_WORKER_TREE;

	FPRINTF(stderr, "ncpus %u \n", ncpus);

	double start_time;
	double end_time;

	start_time = starpu_timing_now();

	co->init(co);

	end_time = starpu_timing_now();

	double timing = (end_time - start_time) / 1000;

	unsigned i;
	for(i = 0; i < ncpus; i++)
	{
		int added = co->add(co, procs[i]);
		FPRINTF(stderr, "added proc %d to the tree \n", added);
	}

	struct starpu_sched_ctx_iterator it;

	int pu;
	co->init_iterator(co, &it);
	while(co->has_next(co, &it))
	{
		pu = co->get_next(co, &it);
		FPRINTF(stderr, "pu = %d out of %u workers \n", pu, co->nworkers);
	}

	unsigned six = 6;
	if (six < ncpus)
		six = ncpus/2;
	for(i = 0; i < six; i++)
	{
		co->remove(co, i);
		FPRINTF(stderr, "remove %u out of %u workers\n", i, co->nworkers);
	}

	while(co->has_next(co, &it))
	{
		pu = co->get_next(co, &it);
		FPRINTF(stderr, "pu = %d out of %u workers \n", pu, co->nworkers);
	}

	FPRINTF(stderr, "timing init = %lf \n", timing);

	co->deinit(co);
	starpu_shutdown();
	free(co);

	return 0;
}
#endif
