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
#include <starpu.h>

#define NITER	10

static void dummy_func(void *descr[] __attribute__ ((unused)), void *arg __attribute__ ((unused)))
{
}

static starpu_codelet dummy_codelet =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dummy_func,
	.cuda_func = dummy_func,
	.nbuffers = 0
};

static create_dummy_task(starpu_tag_t tag)
{
	struct starpu_task *task = starpu_task_create();

	task->use_tag = 1;
	task->tag_id = tag;
	task->cl = &dummy_codelet;
	
	int ret = starpu_submit_task(task);
	if (ret)
	{
		fprintf(stderr, "Warning, no worker can execute the tasks\n");
		/* This is not a bug from StarPU so we return a valid value. */
		exit(0);
	}
}

int main(int argc, char **argv)
{
	starpu_init(NULL);

	starpu_tag_t sync_tags[NITER];

	unsigned iter;
	for (iter = 0; iter < NITER; iter++)
	{
		starpu_tag_t sync_tag = (starpu_tag_t)iter*100;

		sync_tags[iter] = sync_tag;

		unsigned ndeps = 10;
		starpu_tag_t deps[ndeps];

		unsigned d;
		for (d = 0; d < ndeps; d++)
		{
			deps[d] = sync_tag + d + 1; 

			create_dummy_task(deps[d]);
		}

		starpu_create_sync_task(sync_tag, ndeps, deps, NULL, NULL);
	}

	/* Wait all the synchronization tasks */
	starpu_tag_wait_array(NITER, sync_tags);

	starpu_shutdown();

	return 0;
}
