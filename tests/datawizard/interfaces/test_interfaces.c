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

/* XXX Why cant we dereference a handle without this one ? */
#include <core/sched_policy.h>

#include <assert.h>

#include "test_interfaces.h"

/*
 * This is definitely note thrad-safe.
 */
static struct test_config *current_config;

/* TODO :
- OpenCL to OpenCL support
- RAM to RAM ?
*/

/*
 * Users do not know about this enum. They only know that SUCCESS is 0, and
 * FAILURE is 1. Therefore, the values of SUCCESS and FAILURE shall not be
 * changed.
 */
enum exit_code {
	SUCCESS                 = 0,
	FAILURE                 = 1,
	UNTESTED                = 2,
	TASK_CREATION_FAILURE   = 3,
	TASK_SUBMISSION_FAILURE = 4
};

static char *
enum_to_string(exit_code)
{
	switch (exit_code)
	{
		case SUCCESS:
			return "Success";
		case FAILURE:
			return "Failure";
		case UNTESTED:
			return "Untested";
		case TASK_CREATION_FAILURE:
			return "Task creation failed";
		case TASK_SUBMISSION_FAILURE:
			return "Task submission failed";
		default:
			assert(0);
	}
}

struct data_interface_test_summary {
	int success;
#ifdef STARPU_USE_CUDA
	int cpu_to_cuda;
	int cuda_to_cuda;
	int cuda_to_cpu;
	int cpu_to_cuda_async;
	int cuda_to_cpu_async;
	int cuda_to_cuda_async;
#endif
#ifdef STARPU_USE_OPENCL
	int cpu_to_opencl;
	int opencl_to_cpu;
	int cpu_to_opencl_async;
	int opencl_to_cpu_async;
#endif
};

void data_interface_test_summary_print(FILE *f,
				       struct data_interface_test_summary *s)
{
	if (!f)
		f = stderr;

	(void) fprintf(f, "%s : %s\n",
			current_config->name, enum_to_string(s->success));

	(void) fprintf(f, "Asynchronous :\n");
#ifdef STARPU_USE_CUDA
	(void) fprintf(f, "\tCPU    -> CUDA   : %s\n",
			enum_to_string(s->cpu_to_cuda_async));
	(void) fprintf(f, "\tCUDA   -> CUDA   : %s\n",
			enum_to_string(s->cuda_to_cuda_async));
	(void) fprintf(f, "\tCUDA   -> CPU    : %s\n",
			enum_to_string(s->cuda_to_cpu_async));
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	(void) fprintf(f, "\tCPU    -> OpenCl : %s\n",
			enum_to_string(s->cpu_to_opencl_async));
	(void) fprintf(f, "\tOpenCl -< CPU    : %s\n",
			enum_to_string(s->opencl_to_cpu_async));
#endif /* !STARPU_USE_OPENCL */

	(void) fprintf(f, "Synchronous :\n");
#ifdef STARPU_USE_CUDA
	(void) fprintf(f, "\tCPU    -> CUDA   ; %s\n",
			enum_to_string(s->cpu_to_cuda));
	(void) fprintf(f, "\tCUDA   -> CUDA   : %s\n",
			enum_to_string(s->cuda_to_cuda));
	(void) fprintf(f, "\tCUDA   -> CPU    : %s\n",
			enum_to_string(s->cuda_to_cpu));
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	(void) fprintf(f, "\tCPU    -> OpenCl : %s\n",
			enum_to_string(s->cpu_to_opencl));
	(void) fprintf(f, "\tOpenCl -< CPU    : %s\n",
			enum_to_string(s->opencl_to_cpu));
#endif /* !STARPU_USE_OPENCL */
}

int
data_interface_test_summary_success(data_interface_test_summary *s)
{
	return s->success;
}

enum operation {
	CPU_TO_CPU,
#ifdef STARPU_USE_CUDA
	CPU_TO_CUDA,
	CUDA_TO_CUDA,
	CUDA_TO_CPU,
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	CPU_TO_OPENCL,
	OPENCL_TO_CPU
#endif /* !STARPU_USE_OPENCL */
};

static int*
get_field(struct data_interface_test_summary *s, int async, enum operation op)
{
	switch (op)
	{
#ifdef STARPU_USE_CUDA
	case CPU_TO_CUDA:
		return async?&s->cpu_to_cuda_async:&s->cpu_to_cuda;
	case CUDA_TO_CUDA:
		return async?&s->cuda_to_cuda_async:&s->cuda_to_cuda;
	case CUDA_TO_CPU:
		return async?&s->cuda_to_cpu_async:&s->cuda_to_cpu;
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	case CPU_TO_OPENCL:
		return async?&s->cpu_to_opencl_async:&s->cpu_to_opencl;
	case OPENCL_TO_CPU:
		return async?&s->opencl_to_cpu_async:&s->opencl_to_cpu;
#endif /* !STARPU_USE_OPENCL */
		default:
			assert(0);
	}
}

static void
set_field(struct data_interface_test_summary *s, int async,
	  enum operation op, int ret)
{
	int *field = get_field(s, async, op);
	switch (ret)
	{
		case SUCCESS:
			*field = SUCCESS;
			break;
		case FAILURE:
			*field = FAILURE;
			s->success = FAILURE;
			break;
		case UNTESTED:
			*field = UNTESTED;
			break;
		case TASK_CREATION_FAILURE:
			*field = TASK_CREATION_FAILURE;
			break;
		case TASK_SUBMISSION_FAILURE:
			*field = TASK_SUBMISSION_FAILURE;
			break;
		default:
			assert(0);
	}
}

static struct data_interface_test_summary summary = {
#ifdef STARPU_USE_CUDA
	.cpu_to_cuda           = UNTESTED,
	.cuda_to_cuda          = UNTESTED,
	.cuda_to_cpu           = UNTESTED,
	.cpu_to_cuda_async     = UNTESTED,
	.cuda_to_cpu_async     = UNTESTED,
	.cuda_to_cuda_async    = UNTESTED,
#endif
#ifdef STARPU_USE_OPENCL
	.cpu_to_opencl         = UNTESTED,
	.opencl_to_cpu         = UNTESTED,
	.cpu_to_opencl_async   = UNTESTED,
	.opencl_to_cpu_async   = UNTESTED,
#endif
	.success               = SUCCESS
};


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
		case STARPU_GORDON_WORKER: /* Not supported */
		default:
			return -ENODEV;
	}


	struct starpu_task *task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &cl;
	task->buffers[0].handle = *current_config->handle;
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
static enum exit_code
ram_to_cuda(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CUDA_WORKER, 0);
	if (err != 0)
		return TASK_CREATION_FAILURE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}

#if HAVE_CUDA_MEMCPY_PEER
static enum exit_code
cuda_to_cuda(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CUDA_WORKER, 1);
	if (err != 0)
		return TASK_CREATION_FAILURE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}
#endif

static enum exit_code
cuda_to_ram(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
		return TASK_CREATION_FAILURE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static enum exit_code
ram_to_opencl()
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_OPENCL_WORKER, 0);
	if (err != 0)
		return TASK_CREATION_FAILURE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static enum exit_code
opencl_to_ram()
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
		return TASK_CREATION_FAILURE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	fprintf(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}
#endif /* !STARPU_USE_OPENCL */
/* End of the <device1>_to_<device2> functions. */

static void
run_cuda(int async)
{
	/* RAM -> CUDA (-> CUDA) -> RAM */
	int err;
#ifdef STARPU_USE_CUDA
	err = ram_to_cuda();
	set_field(&summary, async, CPU_TO_CUDA, err);
	/* If this failed, there is no point in continuing. */
	if (err != SUCCESS)
		return;

#ifdef HAVE_CUDA_MEMCPY_PEER
	err = cuda_to_cuda();
	set_field(&summary, async, CUDA_TO_CUDA, err);
	/* Even if cuda_to_cuda() failed, a valid copy is left on the first
	 * cuda device, which means we can safely test cuda_to_ram() */
#else
	summary.cuda_to_cuda_async = UNTESTED;
#endif /* !HAVE_CUDA_MEMCPY_PEER */

	err = cuda_to_ram();
	set_field(&summary, async, CUDA_TO_CPU, err);
#endif /* !STARPU_USE_CUDA */
}

static void
run_opencl(int async)
{
	/* RAM -> OpenCL -> RAM */
	int err;

#if STARPU_USE_OPENCL
	err = ram_to_opencl();
	set_field(&summary, async, CPU_TO_OPENCL, err);
	if (err != SUCCESS)
		return;

	err = opencl_to_ram();
	set_field(&summary, async, OPENCL_TO_CPU, err);
#endif /* !STARPU_USE_OPENCL */
}

static void
run_async(void)
{
	run_cuda(1);
	run_opencl(1);
}

static void
run_sync(void)
{
	starpu_data_handle handle = *current_config->handle;

	struct starpu_data_interface_ops *ops = handle->ops;
	//struct starpu_data_copy_methods *copy_methods = ops->copy_methods;
	
//	copy_methods->ram_to_cuda_async = NULL;
	struct starpu_data_interface_ops *new_ops;
	struct starpu_data_copy_methods new_copy_methods;
	memcpy(&new_copy_methods,
		handle->ops->copy_methods,
		sizeof(struct starpu_data_copy_methods));
#ifdef STARPU_USE_CUDA
	new_copy_methods.ram_to_cuda_async = NULL;
	new_copy_methods.cuda_to_cuda_async = NULL;
	new_copy_methods.cuda_to_ram_async = NULL;
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	new_copy_methods.ram_to_opencl_async = NULL;
	new_copy_methods.opencl_to_ram_async = NULL;
#endif /* !STARPU_USE_OPENCL */
	handle->ops->copy_methods = &new_copy_methods;
	run_cuda(0);
	run_opencl(0);
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
	    !config->handle)
	{
		return 1;
	}

	current_config = config;
	return 0;
}

data_interface_test_summary*
run_tests(struct test_config *conf)
{
	if (load_conf(conf) == 1)
	{
		fprintf(stderr, "Failed to load conf.\n");
		return NULL;
	}
	run_async();
	run_sync();
	return &summary;
}
