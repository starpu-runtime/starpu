/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
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

#include <starpu.h>
#include "../common/helper.h"

#ifdef STARPU_USE_CPU
static void dummy_func(void *descr[], void *arg)
{
}

static starpu_codelet cuda_only_cl = 
{
	.where = STARPU_CUDA,
	.cuda_func = dummy_func,
	.model = NULL,
	.nbuffers = 0
};
#endif

int main(int argc, char **argv)
{
#ifdef STARPU_USE_CPU
	int ret;

	/* We force StarPU to use 1 CPU only */
	struct starpu_conf conf;
	memset(&conf, 0, sizeof(conf));
	conf.ncpus = 1;

	ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_task *task = starpu_task_create();
	task->cl = &cuda_only_cl;

	/* Only a CUDA device could execute that task ! */
	ret = starpu_task_submit(task);
	STARPU_ASSERT(ret == -ENODEV);

	struct starpu_task *task_specific = starpu_task_create();
	task_specific->cl = &cuda_only_cl;
	task_specific->execute_on_a_specific_worker = 1;
	task_specific->workerid = 0;

	/* Only a CUDA device could execute that task ! */
	ret = starpu_task_submit(task_specific);
	STARPU_ASSERT(ret == -ENODEV);

	starpu_shutdown();

	return 0;
#else
	fprintf(stderr,"WARNING: Can not test this without CPUs\n");
	return 77;
#endif
}
