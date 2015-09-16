/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2015  Université de Bordeaux
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
#pragma weak smpi_main
extern int smpi_main(int (*realmain) (int argc, char *argv[]), int argc, char *argv[]);
#pragma weak _starpu_mpi_simgrid_init
extern int _starpu_mpi_simgrid_init(int argc, char *argv[]);

struct main_args
{
	int argc;
	char **argv;
};

int do_starpu_main(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[])
{
	struct main_args *args = (void*) argv;
	return starpu_main(args->argc, args->argv);
}

/* In case the MPI application didn't use smpicc to build the file containing
 * main(), try to cope by calling starpu_main */
#pragma weak smpi_simulated_main_
int smpi_simulated_main_(int argc, char *argv[])
{
	if (!starpu_main)
	{
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h or starpu_simgrid_wrap.h included, to properly rename it into starpu_main\n");
		exit(EXIT_FAILURE);
	}

	return starpu_main(argc, argv);
}

#ifdef HAVE_MSG_ENVIRONMENT_GET_ROUTING_ROOT
#ifdef HAVE_MSG_GET_AS_BY_NAME
msg_as_t _starpu_simgrid_get_as_by_name(const char *name)
{
	return MSG_get_as_by_name(name);
}
#else /* HAVE_MSG_GET_AS_BY_NAME */
static msg_as_t __starpu_simgrid_get_as_by_name(msg_as_t root, const char *name)
{
	xbt_dict_t dict;
	xbt_dict_cursor_t cursor;
	const char *key;
	msg_as_t as, ret;
	dict = MSG_environment_as_get_routing_sons(root);
	xbt_dict_foreach(dict, cursor, key, as)
	{
		if (!strcmp(MSG_environment_as_get_name(as), name))
			return as;
		ret = __starpu_simgrid_get_as_by_name(as, name);
		if (ret)
			return ret;
	}
	return NULL;
}

msg_as_t _starpu_simgrid_get_as_by_name(const char *name)
{
	return __starpu_simgrid_get_as_by_name(MSG_environment_get_routing_root(), name);
}
#endif /* HAVE_MSG_GET_AS_BY_NAME */
#endif /* HAVE_MSG_ENVIRONMENT_GET_ROUTING_ROOT */

int _starpu_simgrid_get_nbhosts(const char *prefix)
{
	int ret;
	xbt_dynar_t hosts;
	unsigned i, nb;
	unsigned len = strlen(prefix);
#ifdef HAVE_MSG_ENVIRONMENT_GET_ROUTING_ROOT
	char new_prefix[32];

	if (_starpu_simgrid_running_smpi())
	{
		char name[32];
		STARPU_ASSERT(starpu_mpi_world_rank);
		snprintf(name, sizeof(name), STARPU_MPI_AS_PREFIX"%u", starpu_mpi_world_rank());
		hosts = MSG_environment_as_get_hosts(_starpu_simgrid_get_as_by_name(name));
		len = snprintf(new_prefix, sizeof(new_prefix), "%s-%s", name, prefix);
		prefix = new_prefix;
		len = strlen(prefix);
	}
	else
#endif /* HAVE_MSG_ENVIRONMENT_GET_ROUTING_ROOT */
		hosts = MSG_hosts_as_dynar();
	nb = xbt_dynar_length(hosts);

	ret = 0;
	for (i = 0; i < nb; i++)
	{
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
	char name[32];
	msg_host_t host;
	const char *memsize;

	snprintf(name, sizeof(name), "%s%u", prefix, devid);

	host = _starpu_simgrid_get_host_by_name(name);
	if (!host)
		return 0;

	if (!MSG_host_get_properties(host))
		return 0;

	memsize = MSG_host_get_property_value(host, "memsize");
	if (!memsize)
		return 0;

	return atoll(memsize);
}

msg_host_t _starpu_simgrid_get_host_by_name(const char *name)
{
	if (_starpu_simgrid_running_smpi())
	{
		char mpiname[32];
		STARPU_ASSERT(starpu_mpi_world_rank);
		snprintf(mpiname, sizeof(mpiname), STARPU_MPI_AS_PREFIX"%d-%s", starpu_mpi_world_rank(), name);
		return MSG_get_host_by_name(mpiname);
	}
	else
		return MSG_get_host_by_name(name);
}

msg_host_t _starpu_simgrid_get_host_by_worker(struct _starpu_worker *worker)
{
	char *prefix;
	char name[16];
	msg_host_t host;
	switch (worker->arch)
	{
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
	host =  _starpu_simgrid_get_host_by_name(name);
	STARPU_ASSERT_MSG(host, "Could not find host %s!", name);
	return host;
}

#ifdef STARPU_DEVEL
#warning TODO: use another way to start main, when simgrid provides it, and then include the application-provided configuration for platform numbers
#endif
#undef main
int main(int argc, char **argv)
{
	char path[256];

	if (!starpu_main && !(smpi_main && smpi_simulated_main_))
	{
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h included, to properly rename it into starpu_main\n");
		exit(EXIT_FAILURE);
	}

	if (_starpu_simgrid_running_smpi())
	{
		/* Oops, we are running SMPI, let it start Simgrid, and we'll
		 * take back hand in _starpu_simgrid_init from starpu_init() */
		return smpi_main(_starpu_mpi_simgrid_init, argc, argv);
	}

	MSG_init(&argc, argv);
#if SIMGRID_VERSION_MAJOR < 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR < 9)
	/* Versions earlier than 3.9 didn't support our communication tasks */
	MSG_config("workstation/model", "ptask_L07");
#endif
	/* Simgrid uses tiny stacks by default.  This comes unexpected to our users.  */
	extern xbt_cfg_t _sg_cfg_set;
	unsigned stack_size = 8192;
	struct rlimit rlim;
	if (getrlimit(RLIMIT_STACK, &rlim) == 0 && rlim.rlim_cur != 0 && rlim.rlim_cur != RLIM_INFINITY)
		stack_size = rlim.rlim_cur / 1024;

	xbt_cfg_set_int(_sg_cfg_set, "contexts/stack_size", stack_size);

	/* Load XML platform */
	_starpu_simgrid_get_platform_path(path, sizeof(path));
	MSG_create_environment(path);

	struct main_args *args = malloc(sizeof(*args));
	args->argc = argc;
	args->argv = argv;
	MSG_process_create_with_arguments("main", &do_starpu_main, calloc(MAX_TSD, sizeof(void*)), MSG_get_host_by_name("MAIN"), 0, (char**) args);

	MSG_main();
	return 0;
}

void _starpu_simgrid_init()
{
	if (!starpu_main && !(smpi_main && smpi_simulated_main_))
	{
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h included, to properly rename it into starpu_main\n");
		exit(EXIT_FAILURE);
	}
	if (_starpu_simgrid_running_smpi())
	{
		MSG_process_set_data(MSG_process_self(), calloc(MAX_TSD, sizeof(void*)));
	}
}

/*
 * Tasks
 */

struct task
{
	msg_task_t task;
	int workerid;

	/* communication termination signalization */
	unsigned *finished;
	starpu_pthread_mutex_t *mutex;
	starpu_pthread_cond_t *cond;

	/* Task which waits for this task */
	struct task *next;
};

static struct task *last_task[STARPU_NMAXWORKERS];

/* Actually execute the task.  */
static int task_execute(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[])
{
	struct task *task = (void*) argv;
	_STARPU_DEBUG("task %p started\n", task);
	MSG_task_execute(task->task);
	MSG_task_destroy(task->task);
	_STARPU_DEBUG("task %p finished\n", task);
	STARPU_PTHREAD_MUTEX_LOCK(task->mutex);
	*task->finished = 1;
	STARPU_PTHREAD_COND_BROADCAST(task->cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(task->mutex);

	/* The worker which started this task may be sleeping out of tasks, wake it  */
	starpu_wake_worker(task->workerid);

	if (last_task[task->workerid] == task)
		last_task[task->workerid] = NULL;
	if (task->next)
		MSG_process_create_with_arguments("task", task_execute, calloc(MAX_TSD, sizeof(void*)), MSG_host_self(), 0, (char**) task->next);
	/* Task is freed with process context */
	return 0;
}

/* Wait for completion of all asynchronous tasks for this worker */
void _starpu_simgrid_wait_tasks(int workerid)
{
	struct task *task = last_task[workerid];
	if (!task)
		return;

	unsigned *finished = task->finished;
	starpu_pthread_mutex_t *mutex = task->mutex;
	starpu_pthread_cond_t *cond = task->cond;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	while (!*finished)
		STARPU_PTHREAD_COND_WAIT(cond, mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
}

/* Task execution submitted by StarPU */
void _starpu_simgrid_submit_job(int workerid, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch, double length, unsigned *finished, starpu_pthread_mutex_t *mutex, starpu_pthread_cond_t *cond)
{
	struct starpu_task *starpu_task = j->task;
	msg_task_t simgrid_task;

	if (j->internal)
		/* This is not useful to include in simulation (and probably
		 * doesn't have a perfmodel anyway) */
		return;

	if (isnan(length))
	{
		length = starpu_task_expected_length(starpu_task, perf_arch, j->nimpl);
		STARPU_ASSERT_MSG(!_STARPU_IS_ZERO(length) && !isnan(length),
				"Codelet %s does not have a perfmodel, or is not calibrated enough, please re-run in non-simgrid mode until it is calibrated",
			_starpu_job_get_model_name(j));
	}

	simgrid_task = MSG_task_create(_starpu_job_get_task_name(j),
			length/1000000.0*MSG_get_host_speed(MSG_host_self()),
			0, NULL);

	if (finished == NULL)
	{
		/* Synchronous execution */
		/* First wait for previous tasks */
		_starpu_simgrid_wait_tasks(workerid);
		MSG_task_execute(simgrid_task);
		MSG_task_destroy(simgrid_task);
	}
	else
	{
		/* Asynchronous execution */
		struct task *task = malloc(sizeof(*task));
		task->task = simgrid_task;
		task->workerid = workerid;
		task->finished = finished;
		*finished = 0;
		task->mutex = mutex;
		task->cond = cond;
		task->next = NULL;
		/* Sleep 10µs for the GPU task queueing */
		if (_starpu_simgrid_queue_malloc_cost())
			MSG_process_sleep(0.000010);
		if (last_task[workerid])
		{
			/* Make this task depend on the previous */
			last_task[workerid]->next = task;
			last_task[workerid] = task;
		}
		else
		{
			last_task[workerid] = task;
			MSG_process_create_with_arguments("task", task_execute, calloc(MAX_TSD, sizeof(void*)), MSG_host_self(), 0, (char**) task);
		}
	}
}

/*
 * Transfers
 */

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
static int transfer_execute(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[])
{
	struct transfer *transfer = (void*) argv;
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
			MSG_process_create_with_arguments("transfer task", transfer_execute, calloc(MAX_TSD, sizeof(void*)), _starpu_simgrid_get_host_by_name("MAIN"), 0, (char**) wake);
		}
	}

	free(transfer->wake);
	transfer_list_erase(&pending, transfer);
	/* transfer is freed with process context */
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
		MSG_process_create_with_arguments("transfer task", transfer_execute, calloc(MAX_TSD, sizeof(void*)), _starpu_simgrid_get_host_by_name("MAIN"), 0, (char**) transfer);
	}
}

/* Data transfer issued by StarPU */
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req)
{
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
	if (_starpu_simgrid_queue_malloc_cost())
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
	struct _starpu_pthread_args *_args = (void*) argv;
	struct _starpu_pthread_args args = *_args;
	/* _args is freed with process context */
	args.f(args.arg);
	return 0;
}
#endif
