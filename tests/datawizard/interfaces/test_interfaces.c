/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* XXX Why cant we dereference a handle without this one ? */
#include <core/sched_policy.h>

#include <assert.h>

#include "test_interfaces.h"
#include "../../helper.h"

/*
 * This is definitely note thread-safe.
 */
static struct test_config *current_config;

/* TODO :
- OpenCL to OpenCL support
*/

static char *enum_to_string(int exit_code)
{
	switch (exit_code)
	{
		case SUCCESS:
			return "Success";
		case FAILURE:
			return "Failure";
		case UNTESTED:
			return "Untested";
		case NO_DEVICE:
			return "No device available";
		case TASK_SUBMISSION_FAILURE:
			return "Task submission failed";
		default:
			assert(0);
	}
}

void data_interface_test_summary_print(FILE *f, struct data_interface_test_summary *s)
{
	if (!f)
		f = stderr;

	FPRINTF(f, "%s : %s\n", current_config->name, enum_to_string(s->success));
	FPRINTF(f, "Asynchronous :\n");
	FPRINTF(f, "\tCPU    -> CUDA   : %s\n", enum_to_string(s->cpu_to_cuda_async));
	FPRINTF(f, "\tCUDA   -> CUDA   : %s\n", enum_to_string(s->cuda_to_cuda_async));
	FPRINTF(f, "\tCUDA   -> CPU    : %s\n", enum_to_string(s->cuda_to_cpu_async));
	FPRINTF(f, "\n");
	FPRINTF(f, "\tCPU    -> OpenCL : %s\n", enum_to_string(s->cpu_to_opencl_async));
	FPRINTF(f, "\tOpenCL -> CPU    : %s\n", enum_to_string(s->opencl_to_cpu_async));
	FPRINTF(f, "\n");
	FPRINTF(f, "\tCPU    -> MIC    : %s\n", enum_to_string(s->cpu_to_mic_async));
	FPRINTF(f, "\tMIC    -> CPU    : %s\n", enum_to_string(s->mic_to_cpu_async));
	FPRINTF(f, "\n");

	FPRINTF(f, "Synchronous :\n");
	FPRINTF(f, "\tCPU    -> CUDA   : %s\n", enum_to_string(s->cpu_to_cuda));
	FPRINTF(f, "\tCUDA   -> CUDA   : %s\n", enum_to_string(s->cuda_to_cuda));
	FPRINTF(f, "\tCUDA   -> CPU    : %s\n", enum_to_string(s->cuda_to_cpu));
	FPRINTF(f, "\n");
	FPRINTF(f, "\tCPU    -> OpenCL : %s\n", enum_to_string(s->cpu_to_opencl));
	FPRINTF(f, "\tOpenCL -> CPU    : %s\n", enum_to_string(s->opencl_to_cpu));
	FPRINTF(f, "\n");
	FPRINTF(f, "\tCPU    -> MIC    : %s\n", enum_to_string(s->cpu_to_mic));
	FPRINTF(f, "\tMIC    -> CPU    : %s\n",	enum_to_string(s->mic_to_cpu));

	FPRINTF(f, "\n");
	FPRINTF(f, "CPU -> CPU          : %s\n", enum_to_string(s->cpu_to_cpu));
	FPRINTF(f, "to_pointer()        : %s\n", enum_to_string(s->to_pointer));
	FPRINTF(f, "pointer_is_inside() : %s\n", enum_to_string(s->pointer_is_inside));
	FPRINTF(f, "compare()           : %s\n", enum_to_string(s->compare));
	FPRINTF(f, "pack_unpack()       : %s\n", enum_to_string(s->pack));
}

int data_interface_test_summary_success(struct data_interface_test_summary *s)
{
	return s->success;
}

static void set_field(struct data_interface_test_summary *s, int *field, int ret)
{
	*field = ret;
	if (ret == FAILURE) s->success = ret;
}

static void summary_init(struct data_interface_test_summary *s)
{
	s->cpu_to_cpu            = UNTESTED;
	s->compare               = UNTESTED;
	s->cpu_to_cuda           = UNTESTED;
	s->cuda_to_cuda          = UNTESTED;
	s->cuda_to_cpu           = UNTESTED;
	s->cpu_to_cuda_async     = UNTESTED;
	s->cuda_to_cpu_async     = UNTESTED;
	s->cuda_to_cuda_async    = UNTESTED;
	s->cpu_to_opencl         = UNTESTED;
	s->opencl_to_cpu         = UNTESTED;
	s->cpu_to_opencl_async   = UNTESTED;
	s->opencl_to_cpu_async   = UNTESTED;
	s->cpu_to_mic            = UNTESTED;
	s->mic_to_cpu            = UNTESTED;
	s->cpu_to_mic_async      = UNTESTED;
	s->mic_to_cpu_async      = UNTESTED;
	s->to_pointer            = UNTESTED;
	s->pointer_is_inside     = UNTESTED;
	s->pack                  = UNTESTED;
	s->success               = SUCCESS;
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
 *	- type : STARPU_{CPU,CUDA,OPENCL}_WORKER.
 *      - id: when positive, should be the worker id
 * Return values :
 * 	-ENODEV
 *	0 : success.
 */
static int create_task(struct starpu_task **taskp, enum starpu_worker_archtype type, int id)
{
	static int cpu_workers[STARPU_MAXCPUS];
	static int cuda_workers[STARPU_MAXCUDADEVS];
	static int opencl_workers[STARPU_MAXOPENCLDEVS];
	static int mic_workers[STARPU_MAXMICDEVS];

	static int n_cpus = -1;
	static int n_cudas = -1;
	static int n_opencls = -1;
	static int n_mics = -1;

	if (n_cpus == -1) /* First time here */
	{
		/* We do not check the return values of the calls to
		 * starpu_worker_get_ids_by_type now, because it is simpler to
		 * detect a problem in the switch that comes right after this
		 * block of code. */
		n_cpus = starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, cpu_workers, STARPU_MAXCPUS);
		n_cudas = starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, cuda_workers, STARPU_MAXCUDADEVS);
		n_opencls = starpu_worker_get_ids_by_type(STARPU_OPENCL_WORKER, opencl_workers, STARPU_MAXOPENCLDEVS);
		n_mics = starpu_worker_get_ids_by_type(STARPU_MIC_WORKER, mic_workers, STARPU_MAXMICDEVS);
	}

	int *workers;
	static struct starpu_codelet cl;
	starpu_codelet_init(&cl);
	cl.nbuffers = 1;
	cl.modes[0] = STARPU_RW;

	if (type == STARPU_CPU_WORKER)
	{
		if (n_cpus == 0) return -ENODEV;
		if (id != -1 && id >= n_cpus)
		{
			FPRINTF(stderr, "Not enough CPU workers\n");
			return -ENODEV;
		}
		workers = cpu_workers;
		cl.cpu_funcs[0] = current_config->cpu_func;
	}
	else if (type == STARPU_CUDA_WORKER)
	{
		if (n_cudas == 0) return -ENODEV;
		if (id != -1 && id >= n_cudas)
		{
			FPRINTF(stderr, "Not enough CUDA workers\n");
			return -ENODEV;
		}
		workers = cuda_workers;
		cl.cuda_funcs[0] = current_config->cuda_func;
	}
	else if (type == STARPU_OPENCL_WORKER)
	{
		if (n_opencls == 0) return -ENODEV;
		if (id != -1 && id >= n_opencls)
		{
			FPRINTF(stderr, "Not enough OpenCL workers\n");
			return -ENODEV;
		}
		workers = opencl_workers;
		cl.opencl_funcs[0] = current_config->opencl_func;
	}
	else if (type == STARPU_MIC_WORKER)
	{
		if (n_mics == 0) return -ENODEV;
		if (id != -1 && id >= n_mics)
		{
			FPRINTF(stderr, "Not enough MIC workers\n");
			return -ENODEV;
		}
		workers = mic_workers;
		cl.cpu_funcs_name[0] = current_config->cpu_func_name;
	}
	else
	{
		return -ENODEV;
	}

	factor = -factor;

	struct starpu_task *task;
	task = starpu_task_build(&cl,
				 STARPU_RW, *current_config->handle,
				 STARPU_TASK_SYNCHRONOUS, 1,
				 0);
	task->cl_arg = &factor;
	task->cl_arg_size = sizeof(factor);
	if (id != -1)
	{
		task->execute_on_a_specific_worker = 1;
		task->workerid = workers[id];
	}
	*taskp = task;
	return 0;
}

/*
 * <device1>_to_<device2> functions.
 * They all create and submit a task that has to be executed on <device2>,
 * forcing a copy between <device1> and <device2>.
 */
static enum exit_code ram_to_cuda(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CUDA_WORKER, 0);
	if (err != 0)
		return NO_DEVICE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	FPRINTF(stderr, "[%s] : %d\n", __starpu_func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static enum exit_code cuda_to_cuda(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CUDA_WORKER, 1);
	if (err != 0)
		return NO_DEVICE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	FPRINTF(stderr, "[%s] : %d\n", __starpu_func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static enum exit_code cuda_to_ram(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
		return NO_DEVICE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	FPRINTF(stderr, "[%s] : %d\n", __starpu_func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static enum exit_code ram_to_opencl(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_OPENCL_WORKER, -1);
	if (err != 0)
		return NO_DEVICE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	FPRINTF(stderr, "[%s] : %d\n", __starpu_func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static enum exit_code opencl_to_ram(void)
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
		return NO_DEVICE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	FPRINTF(stderr, "[%s] : %d\n", __starpu_func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static enum exit_code ram_to_mic()
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_MIC_WORKER, -1);
	if (err != 0)
		return NO_DEVICE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	FPRINTF(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}

static enum exit_code mic_to_ram()
{
	int err;
	struct starpu_task *task;

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
		return NO_DEVICE;

	err = starpu_task_submit(task);
	if (err != 0)
		return TASK_SUBMISSION_FAILURE;

	FPRINTF(stderr, "[%s] : %d\n", __func__, current_config->copy_failed);
	return current_config->copy_failed;
}
/* End of the <device1>_to_<device2> functions. */

static void run_cuda(int async, struct data_interface_test_summary *s)
{
	/* RAM -> CUDA (-> CUDA) -> RAM */
	int err;
	err = ram_to_cuda();
	set_field(s, async==1?&s->cpu_to_cuda_async:&s->cpu_to_cuda, err);
	/* If this failed, there is no point in continuing. */
	if (err != SUCCESS)
		return;

	if (starpu_cuda_worker_get_count() >= 2)
	{
		err = cuda_to_cuda();
	}
	else
	{
		err = UNTESTED;
	}
	set_field(s, async==1?&s->cuda_to_cuda_async:&s->cuda_to_cuda, err);
	/* Even if cuda_to_cuda() failed, a valid copy is left on the first
	 * cuda device, which means we can safely test cuda_to_ram() */

	err = cuda_to_ram();
	set_field(s, async==1?&s->cuda_to_cpu_async:&s->cuda_to_cpu, err);
}

static void run_opencl(int async, struct data_interface_test_summary *s)
{
	/* RAM -> OpenCL -> RAM */
	int err;

	err = ram_to_opencl();
	set_field(s, async==1?&s->cpu_to_opencl_async:&s->cpu_to_opencl, err);
	if (err != SUCCESS)
		return;

	err = opencl_to_ram();
	set_field(s, async==1?&s->opencl_to_cpu_async:&s->opencl_to_cpu, err);
}

static void run_mic(int async, struct data_interface_test_summary *s)
{
	int err;

	err = ram_to_mic();
	set_field(s, &s->cpu_to_mic_async, err);
	if (err != SUCCESS)
		return;

	err = mic_to_ram();
	set_field(s, &s->mic_to_cpu_async, err);
}

static void ram_to_ram(struct data_interface_test_summary *s)
{
	int err;
	struct starpu_task *task;
	starpu_data_handle_t src, dst;
	void *src_interface, *dst_interface;

	src = *current_config->handle;
	dst = *current_config->dummy_handle;

	/* We do not care about the nodes */
	src_interface = starpu_data_get_interface_on_node(src, STARPU_MAIN_RAM);
	dst_interface = starpu_data_get_interface_on_node(dst, STARPU_MAIN_RAM);
	if (src->ops->copy_methods->ram_to_ram)
		src->ops->copy_methods->ram_to_ram(src_interface, STARPU_MAIN_RAM, dst_interface, STARPU_MAIN_RAM);
	else
		src->ops->copy_methods->any_to_any(src_interface, STARPU_MAIN_RAM, dst_interface, STARPU_MAIN_RAM, NULL);

	err = create_task(&task, STARPU_CPU_WORKER, -1);
	if (err != 0)
		goto out;

	task->handles[0] = dst;
	err = starpu_task_submit(task);

	if (err != 0)
	{
		err = TASK_SUBMISSION_FAILURE;
		goto out;
	}

	FPRINTF(stderr, "[%s] : %d\n", __starpu_func__, current_config->copy_failed);
	err = current_config->copy_failed;

out:
	set_field(s, &s->cpu_to_cpu, err);
}

static void run_async(struct data_interface_test_summary *s)
{
	int async = starpu_asynchronous_copy_disabled();
	if (async == 1)
	{
		FPRINTF(stderr, "Asynchronous copies have been disabled\n");
		return;
	}
	run_cuda(1, s);
	run_opencl(1, s);
	run_mic(1, s);
}

static void run_sync(struct data_interface_test_summary *s)
{
	starpu_data_handle_t handle = *current_config->handle;
	struct starpu_data_copy_methods new_copy_methods;
	struct starpu_data_copy_methods *old_copy_methods;

	old_copy_methods = (struct starpu_data_copy_methods *) handle->ops->copy_methods;

	memcpy(&new_copy_methods, old_copy_methods, sizeof(struct starpu_data_copy_methods));

	new_copy_methods.ram_to_cuda_async = NULL;
	new_copy_methods.cuda_to_cuda_async = NULL;
	new_copy_methods.cuda_to_ram_async = NULL;
	new_copy_methods.ram_to_opencl_async = NULL;
	new_copy_methods.opencl_to_ram_async = NULL;
	new_copy_methods.ram_to_mic_async = NULL;
	new_copy_methods.mic_to_ram_async = NULL;

	handle->ops->copy_methods = &new_copy_methods;

	run_cuda(0, s);
	run_opencl(0, s);
	run_mic(0, s);

	handle->ops->copy_methods = old_copy_methods;
}

static void compare(struct data_interface_test_summary *s)
{
	int err;
	void *interface_a, *interface_b;
	starpu_data_handle_t handle, dummy_handle;

	handle = *current_config->handle;
	dummy_handle = *current_config->dummy_handle;
	interface_a = starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	interface_b = starpu_data_get_interface_on_node(dummy_handle, STARPU_MAIN_RAM);

	err = handle->ops->compare(interface_a, interface_b);
	s->compare = (err == 0) ? FAILURE : SUCCESS;

	set_field(s, &s->compare, s->compare);
}

static void to_pointer(struct data_interface_test_summary *s)
{
	starpu_data_handle_t handle;

	s->to_pointer = UNTESTED;
	handle = *current_config->handle;
	if (handle->ops->to_pointer)
	{
		unsigned int node;
		unsigned int tests = 0;
		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			if (starpu_node_get_kind(node) != STARPU_CPU_RAM)
				continue;
			if (!starpu_data_test_if_allocated_on_node(handle, node))
				continue;

			void *data_interface = starpu_data_get_interface_on_node(handle, node);
			void *ptr = handle->ops->to_pointer(data_interface, node);
			if (starpu_data_lookup(ptr) != handle)
			{
				s->to_pointer = FAILURE;
				break;
			}
			tests++;
		}
		if (tests > 0)
			s->to_pointer = SUCCESS;
	}
	set_field(s, &s->to_pointer, s->to_pointer);
}

static void pointer_is_inside(struct data_interface_test_summary *s)
{
	starpu_data_handle_t handle;

	s->pointer_is_inside = UNTESTED;
	handle = *current_config->handle;
	if (handle->ops->pointer_is_inside && handle->ops->to_pointer)
	{
		unsigned int node;
		unsigned int tests = 0;

		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			if (starpu_node_get_kind(node) != STARPU_CPU_RAM)
				continue;
			if (!starpu_data_test_if_allocated_on_node(handle, node))
				continue;

			void *data_interface = starpu_data_get_interface_on_node(handle, node);
			void *ptr = handle->ops->to_pointer(data_interface, node);
			if (starpu_data_lookup(ptr) != handle)
			{
				s->pointer_is_inside = FAILURE;
				break;
			}
			if (!starpu_data_pointer_is_inside(handle, node, ptr))
			{
				s->pointer_is_inside = FAILURE;
				break;
			}
			tests++;
		}
		if (tests > 0)
			s->pointer_is_inside = SUCCESS;
	}
	set_field(s, &s->pointer_is_inside, s->pointer_is_inside);
}

static void pack_unpack(struct data_interface_test_summary *s)
{
	starpu_data_handle_t handle;
	starpu_data_handle_t dummy_handle;
	int err = UNTESTED;

	handle = *current_config->handle;
	dummy_handle = *current_config->dummy_handle;
	if (handle->ops->pack_data && handle->ops->unpack_data)
	{
		void *ptr = NULL;
		starpu_ssize_t size = 0;
		starpu_data_pack(handle, &ptr, &size);
		if (size != 0)
		{
			struct starpu_task *task;
			void *mem = (void *)starpu_malloc_on_node_flags(STARPU_MAIN_RAM, size, 0);

			starpu_data_unpack(dummy_handle, mem, size);
			starpu_data_unpack(dummy_handle, ptr, size);

			factor = -factor;
			err = create_task(&task, STARPU_CPU_WORKER, -1);
			if (err != SUCCESS) goto out;

			task->handles[0] = dummy_handle;
			err = starpu_task_submit(task);
			if (err != 0)
			{
				err = TASK_SUBMISSION_FAILURE;
				goto out;
			}

			FPRINTF(stderr, "[%s] : %d\n", __starpu_func__, current_config->copy_failed);
			err = current_config->copy_failed;
		}
	}

 out:
	set_field(s, &s->pack, err);
}

static int load_conf(struct test_config *config)
{
	if (!config ||
#ifdef STARPU_USE_CPU
	    !config->cpu_func ||
	    !config->dummy_handle ||
#endif
#ifdef STARPU_USE_CUDA
	    !config->cuda_func ||
#endif
#ifdef STARPU_USE_OPENCL
	    !config->opencl_func ||
#endif
#ifdef STARPU_USE_MIC
	    !config->cpu_func_name ||
#endif
	    !config->handle)
	{
		return 1;
	}

	current_config = config;
	return 0;
}

void run_tests(struct test_config *conf, struct data_interface_test_summary *s)
{
	summary_init(s);

	if (load_conf(conf) == 1)
	{
		FPRINTF(stderr, "Failed to load conf.\n");
	}

	run_async(s);
	run_sync(s);

	ram_to_ram(s);
	compare(s);
	to_pointer(s);
	pointer_is_inside(s);

	pack_unpack(s);
}
