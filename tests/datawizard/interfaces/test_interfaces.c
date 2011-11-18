/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Institut National de Recherche en Informatique et Automatique
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
#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif

#include "test_interfaces.h"

/* TODO :
- OpenCL to OpenCL support
- RAM to RAM ?
- Asynchronous vs synchronous
- Better error messages
- call starpu_data_unregister
*/

/* Interfaces to test */
extern struct test_config vector_config;

static struct test_config *tests[] = {
	&vector_config,
	NULL
};

static struct test_config *current_config;

/*
 * This variable has to be either -1 or 1.
 * The kernels should check that the ith value stored in the data interface is
 * equal to i, if factor == 1, or -i, if factor == -1.
 */
static int factor = -1;

/*
 * Creates a complete task, only knowing on what device it should be executed.
 * Note that the global variable <current_config> is heavily used here.
 * Arguments :
 *	- taskp : a pointer to a valid task
 *	- type : STARPU_{CPU,CUDA,OPENCL}_WORKER. Gordon is not supported.
 *      - id   : -1 if you dont care about the device where the task will be
 *		 executed, as long as it has the right type.
 *		 >= 0 if you want to make sure the task will be executed on the
 *		 idth device that has the specified type.
 * Return values :
 * 	-ENODEV
 *	0 : success.
 */
static int
create_task(struct starpu_task **taskp, enum starpu_archtype type, int id)
{
	static int cpu_workers[STARPU_MAXCPUS];
	static int cuda_workers[STARPU_MAXCUDADEVS];
	static int opencl_workers[STARPU_MAXOPENCLDEVS];

	static int n_cpus = -1;
	static int n_cudas = -1;
	static int n_opencls = -1;

	if (n_cpus == -1) /* First time here */
	{
		/* XXX Dont check them all at once. */
		/* XXX Error checking */
		n_cpus = starpu_worker_get_ids_by_type(STARPU_CPU_WORKER,
							cpu_workers,
							STARPU_MAXCPUS);
		n_cudas = starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER,
							cuda_workers,
							STARPU_MAXCUDADEVS);
		n_opencls = starpu_worker_get_ids_by_type(STARPU_OPENCL_WORKER,
							opencl_workers,
							STARPU_MAXOPENCLDEVS);
	}

	int workerid;
	static struct starpu_codelet_t cl;
	cl.nbuffers = 1;

	switch (type)
	{
		case STARPU_CPU_WORKER:
			if (id != -1)
			{
				if (id >= n_cpus)
				{
					fprintf(stderr, "Not enough CPU workers\n");
					return -ENODEV;
				}
				workerid = *(cpu_workers + id);
			}
			cl.where = STARPU_CPU;
			cl.cpu_func = current_config->cpu_func;
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_WORKER:
			if (id != -1)
			{
				if (id >= n_cudas)
				{
					fprintf(stderr, "Not enough CUDA workers\n");
					return -ENODEV;
				}
				workerid = cuda_workers[id];
			}
			cl.where = STARPU_CUDA;
			cl.cuda_func = current_config->cuda_func;
			break;
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_WORKER:
			if (id != -1)
			{
				if (id >= n_opencls)
				{
					fprintf(stderr, "Not enough OpenCL workers\n");
					return -ENODEV;
				}
				workerid = *(opencl_workers + id);
			}
			cl.where = STARPU_OPENCL;
			cl.opencl_func = current_config->opencl_func;
			break;
#endif /* ! STARPU_USE_OPENCL */
		default:
			return -ENODEV;
	}


	struct starpu_task *task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &cl;
	task->buffers[0].handle = *(current_config->register_func());
	task->buffers[0].mode = STARPU_RW;
	if (id != -1)
	{
		task->execute_on_a_specific_worker = 1;
		task->workerid = workerid;
	}
	factor = -factor;
	task->cl_arg = &factor;
	task->cl_arg_size = sizeof(&factor);

	*taskp = task;
	return 0;
}

/*
 * <device1>_to_<device2> functions.
 * They all create and submit a task that has to be executed on <device2>,
 * forcing a copy between <device1> and <device2>.
 * XXX : could we sometimes use starp_insert_task() ? It seems hars because we
 * need to set the execute_on_a_specific_worker field...
 */
#ifdef STARPU_USE_CUDA
static int
ram_to_cuda(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CUDA_WORKER, 0);
	if (err != 0)
	{
		fprintf(stderr, "Could not create task\n");
		return 1;
	}

	err = starpu_task_submit(task);
	if (err != 0)
	{
		fprintf(stderr, "Fail : %s\n", strerror(-err));
		return 1;
	}

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static int
cuda_to_cuda(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CUDA_WORKER, 1);
	if (err != 0)
	{
		return 1;
	}

	err = starpu_task_submit(task);
	if (err != 0)
	{
		return 1;
	}

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static int
cuda_to_ram(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
	{
		fprintf(stderr, "Could not create the task\n");
		return 1;
	}

	err = starpu_task_submit(task);
	if (err != 0)
	{
		fprintf(stderr, "Fail : %s\n", strerror(-err));
		return 1;
	}

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static int
ram_to_opencl()
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_OPENCL_WORKER, 0);
	if (err != 0)
	{
		fprintf(stderr, "Could not create the task\n");
		return 1;
	}

	err = starpu_task_submit(task);
	if (err != 0)
	{
		fprintf(stderr, "Fail : %s\n", strerror(-err));
		return 1;
	}

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static int
opencl_to_ram()
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
	{
		fprintf(stderr, "Could not create the task\n");
		return 1;
	}

	err = starpu_task_submit(task);
	if (err != 0)
	{
		fprintf(stderr, "Fail : %s\n", strerror(-err));
		return 1;
	}

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}
#endif /* !STARPU_USE_OPENCL */
/* End of the <device1>_to_<device2> functions. */

static int
run(void)
{
	int err;
#ifdef STARPU_USE_CUDA
	/* RAM -> CUDA -> CUDA -> RAM */
	err = ram_to_cuda();
	if (err != 0)
	{
		fprintf(stderr, "RAM to CUDA failed\n");
		return 1;
	}

#ifdef HAVE_CUDA_MEMCPY_PEER
	err = cuda_to_cuda();
	if (err != 0)
	{
		fprintf(stderr, "CUDA to RAM failed\n");
		return 1;
	}
#endif /* !HAVE_CUDA_MEMCPY_PEER */

	err = cuda_to_ram();
	if (err != 0)
	{
		fprintf(stderr, "CUDA to RAM failed\n");
		return 1;
	}
#endif /* !STARPU_USE_CUDA */

#if STARPU_USE_OPENCL
	/* RAM -> OpenCL -> RAM */
	err = ram_to_opencl();
	if (err != 0)
	{
		fprintf(stderr, "RAM to OpenCL failed\n");
		return 1;
	}

	err = opencl_to_ram();
	if (err != 0)
	{
		fprintf(stderr, "OpenCL to RAM failed\n");
		return 1;
	}
#endif /* !STARPU_USE_OPENCL */

	return 0;
}

static int
load_conf(struct test_config *config)
{
	if (!config ||
	    !config->cpu_func ||
#ifdef STARPU_USE_CUDA
	    !config->cuda_func ||
#endif
#ifdef STARPU_USE_OPENCL
	    !config->opencl_func ||
#endif
	    !config->register_func)
	{
		return 1;
	}

	current_config = config;
	return 0;
}

int
main(void)
{
	int i;
	int err;

	err = starpu_init(NULL);
	if (err != 0)
	{
		fprintf(stderr, "starpu_init failed, not running the tests\n");
		return EXIT_FAILURE;
	}

	for (i = 0; tests[i] != NULL; ++i)
	{
		err = load_conf(tests[i]);
		if (err != 0)
		{
			fprintf(stderr, "Skipping test, invalid conf\n");
			continue;
		}

		err = run();
		if (err != 0)
			fprintf(stderr, "%s : FAIL\n", current_config->name);
		else
			fprintf(stderr, "%s : OK\n", current_config->name);
	}

	starpu_shutdown();
	return EXIT_SUCCESS;
}
