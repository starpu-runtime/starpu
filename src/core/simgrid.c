/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2016, 2019  Université de Bordeaux
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
#include <datawizard/memory_nodes.h>
#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <core/perfmodel/perfmodel.h>
#include <core/workers.h>
#include <core/simgrid.h>

#ifdef STARPU_SIMGRID
#include <sys/resource.h>

#pragma weak starpu_main
extern int starpu_main(int argc, char *argv[]);

static int main_ret;

struct main_args
{
	int argc;
	char **argv;
};

int do_starpu_main(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	struct main_args *args = MSG_process_get_data(MSG_process_self());
	main_ret = starpu_main(args->argc, args->argv);
	return main_ret;
}

int _starpu_simgrid_get_nbhosts(const char *prefix)
{
	int ret;
	xbt_dynar_t hosts = MSG_hosts_as_dynar();
	unsigned i, nb = xbt_dynar_length(hosts);
	unsigned len = strlen(prefix);

	ret = 0;
	for (i = 0; i < nb; i++) {
		const char *name;
		name = MSG_host_get_name(xbt_dynar_get_as(hosts, i, msg_host_t));
		if (!strncmp(name, prefix, len))
			ret++;
	}
	xbt_dynar_free(&hosts);
	return ret;
}

unsigned long long _starpu_simgrid_get_memsize(const char *prefix, unsigned devid)
{
	char name[16];
	msg_host_t host;
	const char *memsize;

	snprintf(name, sizeof(name), "%s%u", prefix, devid);

	host = MSG_get_host_by_name(name);
	if (!host)
		return 0;

	if (!MSG_host_get_properties(host))
		return 0;

	memsize = MSG_host_get_property_value(host, "memsize");
	if (!memsize)
		return 0;

	return atoll(memsize);
}

msg_host_t _starpu_simgrid_get_host_by_worker(struct _starpu_worker *worker)
{
	char *prefix;
	char name[16];
	msg_host_t host;
	switch (worker->arch) {
		case STARPU_CPU_WORKER:
			prefix = "CPU";
			break;
		case STARPU_CUDA_WORKER:
			prefix = "CUDA";
			break;
		case STARPU_OPENCL_WORKER:
			prefix = "OpenCL";
			break;
		default:
			STARPU_ASSERT(0);
	}
	snprintf(name, sizeof(name), "%s%d", prefix, worker->devid);
	host =  MSG_get_host_by_name(name);
	STARPU_ASSERT_MSG(host, "Could not find host %s!", name);
	return host;
}

#ifdef STARPU_DEVEL
#warning TODO: use another way to start main, when simgrid provides it, and then include the application-provided configuration for platform numbers
#endif
#undef main
int main(int argc, char **argv)
{
	xbt_dynar_t hosts;
	int i;
	char path[256];

	if (!starpu_main)
	{
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h included, to properly rename it into starpu_main\n");
		exit(EXIT_FAILURE);
	}

	MSG_init(&argc, argv);
#if SIMGRID_VERSION_MAJOR < 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR < 9)
	/* Versions earlier than 3.9 didn't support our communication tasks */
	MSG_config("workstation/model", "ptask_L07");
#endif
	/* Simgrid uses tiny stacks by default.  This comes unexpected to our users.  */
	unsigned stack_size = 8192;
	struct rlimit rlim;
	if (getrlimit(RLIMIT_STACK, &rlim) == 0 && rlim.rlim_cur != 0 && rlim.rlim_cur != RLIM_INFINITY)
		stack_size = rlim.rlim_cur / 1024;

#if SIMGRID_VERSION_MAJOR < 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR < 13)
	extern xbt_cfg_t _sg_cfg_set;
	xbt_cfg_set_int(_sg_cfg_set, "contexts/stack_size", stack_size);
#else
	xbt_cfg_set_int("contexts/stack-size", stack_size);
#endif

	/* Load XML platform */
#if SIMGRID_VERSION_MAJOR < 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR < 13)
	_starpu_simgrid_get_platform_path(3, path, sizeof(path));
#else
	_starpu_simgrid_get_platform_path(4, path, sizeof(path));
#endif
	MSG_create_environment(path);

	hosts = MSG_hosts_as_dynar();
	int nb = xbt_dynar_length(hosts);
	for (i = 0; i < nb; i++)
		MSG_host_set_data(xbt_dynar_get_as(hosts, i, msg_host_t), calloc(MAX_TSD, sizeof(void*)));

	struct main_args args = { .argc = argc, .argv = argv };
	MSG_process_create("main", &do_starpu_main, &args, MSG_get_host_by_name("MAIN"));
	xbt_dynar_free(&hosts);

	MSG_main();
	return main_ret;
}

void _starpu_simgrid_init()
{
	if (!starpu_main)
	{
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h included, to properly rename it into starpu_main\n");
		exit(EXIT_FAILURE);
	}
}

/* Task execution submitted by StarPU */
void _starpu_simgrid_execute_job(struct _starpu_job *j, enum starpu_perfmodel_archtype perf_arch, double length)
{
	struct starpu_task *task = j->task;
	msg_task_t simgrid_task;

	if (j->internal)
		/* This is not useful to include in simulation (and probably
		 * doesn't have a perfmodel anyway) */
		return;

	if (isnan(length))
	{
		length = starpu_task_expected_length(task, perf_arch, j->nimpl);
		STARPU_ASSERT_MSG(!_STARPU_IS_ZERO(length) && !isnan(length),
				"Codelet %s does not have a perfmodel, or is not calibrated enough, please re-run in non-simgrid mode until it is calibrated",
			_starpu_job_get_model_name(j));
	}

	simgrid_task = MSG_task_create(_starpu_job_get_model_name(j),
#ifdef HAVE_MSG_HOST_GET_SPEED
			length/1000000.0*MSG_host_get_speed(MSG_host_self()),
#else
			length/1000000.0*MSG_get_host_speed(MSG_host_self()),
#endif
			0, NULL);
	MSG_task_execute(simgrid_task);
	MSG_task_destroy(simgrid_task);
}

/* Note: simgrid is not parallel, so there is no need to hold locks for management of transfers.  */
LIST_TYPE(transfer,
	msg_task_t task;
	int src_node;
	int dst_node;
	int run_node;

	/* communication termination signalization */
	unsigned *finished;
	starpu_pthread_mutex_t *mutex;
	starpu_pthread_cond_t *cond;

	/* transfers which wait for this transfer */
	struct transfer **wake;
	unsigned nwake;

	/* Number of transfers that this transfer waits for */
	unsigned nwait;
)

struct transfer_list pending;

/* Tell for two transfers whether they should be handled in sequence */
static int transfers_are_sequential(struct transfer *new_transfer, struct transfer *old_transfer)
{
	int new_is_cuda STARPU_ATTRIBUTE_UNUSED, old_is_cuda STARPU_ATTRIBUTE_UNUSED;
	int new_is_opencl STARPU_ATTRIBUTE_UNUSED, old_is_opencl STARPU_ATTRIBUTE_UNUSED;
	int new_is_gpu_gpu, old_is_gpu_gpu;

	new_is_cuda  = starpu_node_get_kind(new_transfer->src_node) == STARPU_CUDA_RAM;
	new_is_cuda |= starpu_node_get_kind(new_transfer->dst_node) == STARPU_CUDA_RAM;
	old_is_cuda  = starpu_node_get_kind(old_transfer->src_node) == STARPU_CUDA_RAM;
	old_is_cuda |= starpu_node_get_kind(old_transfer->dst_node) == STARPU_CUDA_RAM;

	new_is_opencl  = starpu_node_get_kind(new_transfer->src_node) == STARPU_OPENCL_RAM;
	new_is_opencl |= starpu_node_get_kind(new_transfer->dst_node) == STARPU_OPENCL_RAM;
	old_is_opencl  = starpu_node_get_kind(old_transfer->src_node) == STARPU_OPENCL_RAM;
	old_is_opencl |= starpu_node_get_kind(old_transfer->dst_node) == STARPU_OPENCL_RAM;

	new_is_gpu_gpu = new_transfer->src_node && new_transfer->dst_node;
	old_is_gpu_gpu = old_transfer->src_node && old_transfer->dst_node;

	/* We ignore cuda-opencl transfers, they can not happen */
	STARPU_ASSERT(!((new_is_cuda && old_is_opencl) || (old_is_cuda && new_is_opencl)));

	/* The following constraints have been observed with CUDA alone */

	/* Same source/destination, sequential */
	if (new_transfer->src_node == old_transfer->src_node && new_transfer->dst_node == old_transfer->dst_node)
		return 1;

	/* Crossed GPU-GPU, sequential */
	if (new_is_gpu_gpu
			&& new_transfer->src_node == old_transfer->dst_node
			&& old_transfer->src_node == new_transfer->dst_node)
		return 1;

	/* GPU-GPU transfers are sequential with any RAM->GPU transfer */
	if (new_is_gpu_gpu
			&& old_transfer->dst_node == new_transfer->src_node
			&& old_transfer->dst_node == new_transfer->dst_node)
		return 1;
	if (old_is_gpu_gpu
			&& new_transfer->dst_node == old_transfer->src_node
			&& new_transfer->dst_node == old_transfer->dst_node)
		return 1;

	/* StarPU's constraint on CUDA transfers is using one stream per
	 * source/destination pair, which is already handled above */

	return 0;
}

/* Actually execute the transfer, and then start transfers waiting for this one.  */
static int transfer_execute(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	struct transfer *transfer = MSG_process_get_data(MSG_process_self());
	unsigned i;
	_STARPU_DEBUG("transfer %p started\n", transfer);
	MSG_task_execute(transfer->task);
	MSG_task_destroy(transfer->task);
	_STARPU_DEBUG("transfer %p finished\n", transfer);
	STARPU_PTHREAD_MUTEX_LOCK(transfer->mutex);
	*transfer->finished = 1;
	STARPU_PTHREAD_COND_BROADCAST(transfer->cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(transfer->mutex);

	/* The workers which started this request may be sleeping out of tasks, wake it  */
	_starpu_wake_all_blocked_workers_on_node(transfer->run_node);

	/* Wake transfers waiting for my termination */
	/* Note: due to possible preemption inside process_create, the array
	 * may grow while doing this */
	for (i = 0; i < transfer->nwake; i++)
	{
		struct transfer *wake = transfer->wake[i];
		STARPU_ASSERT(wake->nwait > 0);
		wake->nwait--;
		if (!wake->nwait)
		{
			_STARPU_DEBUG("triggering transfer %p\n", wake);
			MSG_process_create("transfer task", transfer_execute, wake, MSG_get_host_by_name("MAIN"));
		}
	}

	free(transfer->wake);
	transfer_list_erase(&pending, transfer);
	transfer_delete(transfer);
	return 0;
}

/* Look for sequentialization between this transfer and pending transfers, and submit this one */
static void transfer_submit(struct transfer *transfer)
{
	struct transfer *old;

	for (old  = transfer_list_begin(&pending);
	     old != transfer_list_end(&pending);
	     old  = transfer_list_next(old))
	{
		if (transfers_are_sequential(transfer, old))
		{
			_STARPU_DEBUG("transfer %p(%d->%d) waits for %p(%d->%d)\n",
					transfer, transfer->src_node, transfer->dst_node,
					old, old->src_node, old->dst_node);
			/* Make new wait for the old */
			transfer->nwait++;
			/* Make old wake the new */
			old->wake = realloc(old->wake, (old->nwake + 1) * sizeof(old->wake));
			old->wake[old->nwake] = transfer;
			old->nwake++;
		}
	}

	transfer_list_push_front(&pending, transfer);

	if (!transfer->nwait)
	{
		_STARPU_DEBUG("transfer %p waits for nobody, starting\n", transfer);
		MSG_process_create("transfer task", transfer_execute, transfer, MSG_get_host_by_name("MAIN"));
	}
}

/* Data transfer issued by StarPU */
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req)
{
	/* Simgrid does not like 0-bytes transfers */
	if (!size)
		return 0;
	msg_task_t task;
	msg_host_t *hosts = calloc(2, sizeof(*hosts));
	double *computation = calloc(2, sizeof(*computation));
	double *communication = calloc(4, sizeof(*communication));
	starpu_pthread_mutex_t mutex;
	starpu_pthread_cond_t cond;
	unsigned finished;

	hosts[0] = _starpu_simgrid_memory_node_get_host(src_node);
	hosts[1] = _starpu_simgrid_memory_node_get_host(dst_node);
	STARPU_ASSERT(hosts[0] != hosts[1]);
	communication[1] = size;

	task = MSG_parallel_task_create("copy", 2, hosts, computation, communication, NULL);

	struct transfer *transfer = transfer_new();

	_STARPU_DEBUG("creating transfer %p for %lu bytes\n", transfer, (unsigned long) size);

	transfer->task = task;
	transfer->src_node = src_node;
	transfer->dst_node = dst_node;
	transfer->run_node = _starpu_memory_node_get_local_key();

	if (req)
	{
		transfer->finished = &req->async_channel.event.finished;
		transfer->mutex = &req->async_channel.event.mutex;
		transfer->cond = &req->async_channel.event.cond;
	}
	else
	{
		transfer->finished = &finished;
		transfer->mutex = &mutex;
		transfer->cond = &cond;
	}

	*transfer->finished = 0;
	STARPU_PTHREAD_MUTEX_INIT(transfer->mutex, NULL);
	STARPU_PTHREAD_COND_INIT(transfer->cond, NULL);
	transfer->wake = NULL;
	transfer->nwake = 0;
	transfer->nwait = 0;

	if (req)
		_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);

	/* Sleep 10µs for the GPU transfer queueing */
	MSG_process_sleep(0.000010);
	transfer_submit(transfer);
	/* Note: from here, transfer might be already freed */

	if (req)
	{
		_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);
		_STARPU_TRACE_DATA_COPY(src_node, dst_node, size);
		return -EAGAIN;
	}
	else
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		while (!finished)
			STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return 0;
	}
}

int
_starpu_simgrid_thread_start(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[])
{
	void *(*f)(void*) = (void*) (uintptr_t) strtol(argv[0], NULL, 16);
	void *arg = (void*) (uintptr_t) strtol(argv[1], NULL, 16);

	/* _args is freed with process context */
	f(arg);
	return 0;
}
#endif
