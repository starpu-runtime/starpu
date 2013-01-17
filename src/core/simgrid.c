/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2013  Université de Bordeaux 1
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
#include <unistd.h>
#include <core/perfmodel/perfmodel.h>
#include <core/workers.h>

#ifdef STARPU_SIMGRID
#include <msg/msg.h>

#define MAX_TSD 16

#pragma weak starpu_main
extern int starpu_main(int argc, char *argv[]);

static struct starpu_conf conf;

struct main_args {
	int argc;
	char **argv;
};

int do_starpu_main(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	struct main_args *args = MSG_process_get_data(MSG_process_self());
	return starpu_main(args->argc, args->argv);
}

static void bus_name(char *s, size_t size, int num)
{
	if (!num)
		snprintf(s, size, "RAM");
	else if (num < conf.ncuda + 1)
		snprintf(s, size, "CUDA%d", num - 1);
	else
		snprintf(s, size, "OpenCL%d", num - conf.ncuda - 1);
}

#ifdef STARPU_DEVEL
#warning TODO: use another way to start main, when simgrid provides it, and then include the application-provided configuration for platform numbers
#endif
#undef main
int main(int argc, char **argv)
{
	xbt_dynar_t hosts;
	int i, j;
	char name[] = "/tmp/starpu-simgrid-platform.xml.XXXXXX";
	int fd;
	FILE *file;
	struct starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;

	if (!starpu_main)
	{
		_STARPU_ERROR("The main file of this application needs to be compiled with starpu.h included, to properly define starpu_main\n");
		exit(EXIT_FAILURE);
	}

	MSG_init(&argc, argv);
#if SIMGRID_VERSION_MAJOR < 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR < 10)
	/* Versions earlier than 3.10 didn't support our communication tasks */
	MSG_config("workstation/model", "ptask_L07");
#endif

	/* Create platform file */
	starpu_conf_init(&conf);
	if ((!getenv("STARPU_NCPUS") && !getenv("STARPU_NCPU"))
#ifdef STARPU_USE_CUDA
	 || !getenv("STARPU_NCUDA")
#endif
#ifdef STARPU_USE_OPENCL
	 || !getenv("STARPU_NOPENCL")
#endif
			)
	{
		_STARPU_ERROR("Please specify the number of cpus and gpus by setting the environment variables STARPU_NCPU%s%s\n",
#ifdef STARPU_USE_CUDA
			      ", STARPU_NCUDA",
#else
			      "",
#endif
#ifdef STARPU_USE_OPENCL
			      ", STARPU_NOPENCL"
#else
			      ""
#endif
			);
		exit(EXIT_FAILURE);
	}
	_starpu_conf_check_environment(&conf);

	_starpu_load_bus_performance_files();

	topology->ncpus = conf.ncpus;
	topology->ncudagpus = conf.ncuda;
	topology->nopenclgpus = conf.nopencl;

	/* TODO: rather use simgrid/platf.h */
	/* TODO: but still permit the user to provide his own xml */
	fd = mkstemp(name);
	file = fdopen(fd, "w");
	fprintf(file,
"<?xml version='1.0'?>\n"
" <!DOCTYPE platform SYSTEM 'http://simgrid.gforge.inria.fr/simgrid.dtd'>\n"
" <platform version='3'>\n"
" <config id='General'>\n"
"   <prop id='network/TCP_gamma' value='-1'></prop>\n"
"   <prop id='network/latency_factor' value='1'></prop>\n"
"   <prop id='network/bandwidth_factor' value='1'></prop>\n"
" </config>\n"
" <AS  id='AS0'  routing='Full'>\n"
"   <host id='MAIN' power='1'/>\n"
		);

	for (i = 0; i < conf.ncpus; i++)
		fprintf(file, "   <host id='CPU%d' power='2000000000'/>\n", i);

	for (i = 0; i < conf.ncuda; i++)
		fprintf(file, "   <host id='CUDA%d' power='2000000000'/>\n", i);

	for (i = 0; i < conf.nopencl; i++)
		fprintf(file, "   <host id='OpenCL%d' power='2000000000'/>\n", i);

	fprintf(file, "\n   <host id='RAM' power='1'/>\n");

	/* Compute maximum bandwidth, taken as machine bandwidth */
	double max_bandwidth = 0;
	for (i = 1; i < conf.ncuda + conf.nopencl + 1; i++)
	{
		if (max_bandwidth < _starpu_transfer_bandwidth(0, i))
			max_bandwidth = _starpu_transfer_bandwidth(0, i);
		if (max_bandwidth < _starpu_transfer_bandwidth(i, 0))
			max_bandwidth = _starpu_transfer_bandwidth(i, 0);
	}
	fprintf(file, "\n   <link id='Share' bandwidth='%f' latency='0.000000'/>\n\n", max_bandwidth*1000000);

	for (i = 0; i < conf.ncuda + conf.nopencl + 1; i++)
	{
		char i_name[16];
		bus_name(i_name, sizeof(i_name), i);

		for (j = 0; j < conf.ncuda + conf.nopencl + 1; j++)
		{
			char j_name[16];
			if (j == i)
				continue;
			bus_name(j_name, sizeof(j_name), j);
			fprintf(file, "   <link id='%s-%s' bandwidth='%f' latency='%f'/>\n",
				i_name, j_name,
				_starpu_transfer_bandwidth(i, j) * 1000000,
				_starpu_transfer_latency(i, j) / 1000000);
		}
	}

	for (i = 0; i < conf.ncuda + conf.nopencl + 1; i++)
	{
		char i_name[16];
		bus_name(i_name, sizeof(i_name), i);

		for (j = 0; j < conf.ncuda + conf.nopencl + 1; j++)
		{
			char j_name[16];
			if (j == i)
				continue;
			bus_name(j_name, sizeof(j_name), j);
			fprintf(file,
"   <route src='%s' dst='%s' symmetrical='NO'><link_ctn id='%s-%s'/><link_ctn id='Share'/></route>\n",
				i_name, j_name, i_name, j_name);
		}
	}

	fprintf(file, 
" </AS>\n"
" </platform>\n"
		);
	fclose(file);
	close(fd);

	/* and load it */
	MSG_create_environment(name);
	unlink(name);

	hosts = MSG_hosts_as_dynar();
	int nb = xbt_dynar_length(hosts);
	for (i = 0; i < nb; i++)
		MSG_host_set_data(xbt_dynar_get_as(hosts, i, msg_host_t), calloc(MAX_TSD, sizeof(void*)));

	struct main_args args = { .argc = argc, .argv = argv };
	MSG_process_create("main", &do_starpu_main, &args, xbt_dynar_get_as(hosts, 0, msg_host_t));
	xbt_dynar_free(&hosts);

	MSG_main();
	return 0;
}

/* Task execution submitted by StarPU */
void _starpu_simgrid_execute_job(struct _starpu_job *j, enum starpu_perf_archtype perf_arch, double length)
{
	struct starpu_task *task = j->task;
	msg_task_t simgrid_task;

	if (j->exclude_from_dag)
		/* This is not useful to include in simulation (and probably
		 * doesn't have a perfmodel anyway) */
		return;

	if (isnan(length))
	{
		length = starpu_task_expected_length(task, perf_arch, j->nimpl);
		STARPU_ASSERT_MSG(!_STARPU_IS_ZERO(length) && !isnan(length),
			"Codelet %s does not have a perfmodel, or is not calibrated enough",
			_starpu_job_get_model_name(j));
	}

	simgrid_task = MSG_task_create(_starpu_job_get_model_name(j),
			length/1000000.0*MSG_get_host_speed(MSG_host_self()),
			0, NULL);
	MSG_task_execute(simgrid_task);
}

/* Note: simgrid is not parallel, so there is no need to hold locks for management of transfers.  */
LIST_TYPE(transfer,
	msg_task_t task;
	int src_node;
	int dst_node;
	int run_node;

	/* communication termination signalization */
	unsigned *finished;
	_starpu_pthread_mutex_t *mutex;
	_starpu_pthread_cond_t *cond;

	/* transfers which wait for this transfer */
	struct transfer **wake;
	unsigned nwake;

	/* Number of transfers that this transfer waits for */
	unsigned nwait;
)

struct transfer_list *pending;

/* Tell for two transfers whether they should be handled in sequence */
static int transfers_are_sequential(struct transfer *new_transfer, struct transfer *old_transfer)
{
	int new_is_cuda STARPU_ATTRIBUTE_UNUSED, old_is_cuda STARPU_ATTRIBUTE_UNUSED;
	int new_is_opencl STARPU_ATTRIBUTE_UNUSED, old_is_opencl STARPU_ATTRIBUTE_UNUSED;
	int new_is_gpu_gpu, old_is_gpu_gpu;

	new_is_cuda  = new_transfer->src_node >= 1 && new_transfer->src_node <= conf.ncuda;
	new_is_cuda |= new_transfer->dst_node >= 1 && new_transfer->dst_node <= conf.ncuda;
	old_is_cuda  = old_transfer->src_node >= 1 && old_transfer->src_node <= conf.ncuda;
	old_is_cuda |= old_transfer->dst_node >= 1 && old_transfer->dst_node <= conf.ncuda;

	new_is_opencl  = new_transfer->src_node > conf.ncuda && new_transfer->src_node <= conf.ncuda + conf.nopencl;
	new_is_opencl |= new_transfer->dst_node > conf.ncuda && new_transfer->dst_node <= conf.ncuda + conf.nopencl;
	old_is_opencl  = old_transfer->src_node > conf.ncuda && old_transfer->src_node <= conf.ncuda + conf.nopencl;
	old_is_opencl |= old_transfer->dst_node > conf.ncuda && old_transfer->dst_node <= conf.ncuda + conf.nopencl;

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

	/* These constraints come from StarPU */

	/* StarPU uses one stream per direction */
	/* RAM->GPU and GPU->RAM are already handled by "same source/destination" */

	/* StarPU uses one stream per running GPU for GPU-GPU transfers */
	if (new_is_gpu_gpu && old_is_gpu_gpu && new_transfer->run_node == old_transfer->run_node)
		return 1;

	return 0;
}

/* Actually execute the transfer, and then start transfers waiting for this one.  */
static int transfer_execute(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	struct transfer *transfer = MSG_process_get_data(MSG_process_self());
	unsigned i;
	MSG_task_execute(transfer->task);
	MSG_task_destroy(transfer->task);
	_STARPU_PTHREAD_MUTEX_LOCK(transfer->mutex);
	*transfer->finished = 1;
	_STARPU_PTHREAD_COND_BROADCAST(transfer->cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(transfer->mutex);

	/* Wake transfers waiting for my termination */
	/* Note: due to possible preemption inside process_create, the array
	 * may grow while doing this */
	for (i = 0; i < transfer->nwake; i++) {
		struct transfer *wake = transfer->wake[i];
		STARPU_ASSERT(wake->nwait > 0);
		wake->nwait--;
		if (!wake->nwait) {
			_STARPU_DEBUG("triggering transfer %p\n", wake);
			MSG_process_create("transfer task", transfer_execute, wake, MSG_get_host_by_name("MAIN"));
		}
	}

	free(transfer->wake);
	transfer_list_erase(pending, transfer);
	transfer_delete(transfer);
	return 0;
}

/* Look for sequentialization between this transfer and pending transfers, and submit this one */
static void transfer_submit(struct transfer *transfer) {
	struct transfer *old;

	if (!pending)
		pending = transfer_list_new();

	for (old  = transfer_list_begin(pending);
	     old != transfer_list_end(pending);
	     old  = transfer_list_next(old)) {
		if (transfers_are_sequential(transfer, old)) {
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

	transfer_list_push_front(pending, transfer);

	if (!transfer->nwait) {
		_STARPU_DEBUG("transfer %p waits for nobody, starting\n", transfer);
		MSG_process_create("transfer task", transfer_execute, transfer, MSG_get_host_by_name("MAIN"));
	}
}

/* Data transfer issued by StarPU */
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req)
{
	msg_task_t task;
	msg_host_t *hosts = calloc(2, sizeof(*hosts));
	double *computation = calloc(2, sizeof(*computation));
	double *communication = calloc(4, sizeof(*communication));
	_starpu_pthread_mutex_t mutex;
	_starpu_pthread_cond_t cond;
	unsigned finished;

	hosts[0] = _starpu_simgrid_memory_node_get_host(src_node);
	hosts[1] = _starpu_simgrid_memory_node_get_host(dst_node);
	STARPU_ASSERT(hosts[0] != hosts[1]);
	communication[1] = size;

	task = MSG_parallel_task_create("copy", 2, hosts, computation, communication, NULL);

	struct transfer *transfer = transfer_new();
	transfer->task = task;
	transfer->src_node = src_node;
	transfer->dst_node = dst_node;
	transfer->run_node = _starpu_get_local_memory_node();

	if (req) {
		transfer->finished = &req->async_channel.event.finished;
		transfer->mutex = &req->async_channel.event.mutex;
		transfer->cond = &req->async_channel.event.cond;
	} else {
		transfer->finished = &finished;
		transfer->mutex = &mutex;
		transfer->cond = &cond;
	}

	*transfer->finished = 0;
	_STARPU_PTHREAD_MUTEX_INIT(transfer->mutex, NULL);
	_STARPU_PTHREAD_COND_INIT(transfer->cond, NULL);
	transfer->wake = NULL;
	transfer->nwake = 0;
	transfer->nwait = 0;

	if (req)
		_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);

	transfer_submit(transfer);
	/* Note: from here, transfer might be already freed */

	if (req) {
		_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);
		_STARPU_TRACE_DATA_COPY(src_node, dst_node, size);
		return -EAGAIN;
	} else {
		/* this is not associated to a request so it's synchronous */
		_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		while (!finished)
			_STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
		_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		return 0;
	}
}

static int last_key;

int _starpu_pthread_key_create(_starpu_pthread_key_t *key)
{
	/* Note: no synchronization here, we are actually monothreaded anyway. */
	STARPU_ASSERT(last_key < MAX_TSD);
	*key = last_key++;
	return 0;
}

int _starpu_pthread_key_delete(_starpu_pthread_key_t key STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_DEVEL
#warning TODO: implement pthread_key_delete so simgridified starpu can be restarted at will
#endif
	return 0;
}

int _starpu_pthread_setspecific(_starpu_pthread_key_t key, void *ptr)
{
	void **array = MSG_host_get_data(MSG_host_self());
	array[key] = ptr;
	return 0;
}

void* _starpu_pthread_getspecific(_starpu_pthread_key_t key)
{
	void **array = MSG_host_get_data(MSG_host_self());
	return array[key];
}

int
_starpu_simgrid_thread_start(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	struct _starpu_pthread_args *args = MSG_process_get_data(MSG_process_self());
	args->f(args->arg);
	free(args);
	return 0;
}
#endif
