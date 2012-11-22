/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Université de Bordeaux 1
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

static void bus_name(struct starpu_conf *conf, char *s, size_t size, int num)
{
	if (!num)
		snprintf(s, size, "RAM");
	else if (num < conf->ncuda + 1)
		snprintf(s, size, "CUDA%d", num - 1);
	else
		snprintf(s, size, "OpenCL%d", num - conf->ncuda - 1);
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
	MSG_config("workstation/model", "ptask_L07");

	/* Create platform file */
	struct starpu_conf conf;
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

	fd = mkstemp(name);
	file = fdopen(fd, "w");
	fprintf(file,
"<?xml version='1.0'?>\n"
" <!DOCTYPE platform SYSTEM 'http://simgrid.gforge.inria.fr/simgrid.dtd'>\n"
" <platform version='3'>\n"
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
		bus_name(&conf, i_name, sizeof(i_name), i);

		for (j = 0; j < conf.ncuda + conf.nopencl + 1; j++)
		{
			char j_name[16];
			if (j == i)
				continue;
			bus_name(&conf, j_name, sizeof(j_name), j);
			fprintf(file, "   <link id='%s-%s' bandwidth='%f' latency='%f'/>\n",
				i_name, j_name,
				_starpu_transfer_bandwidth(i, j) * 1000000,
				_starpu_transfer_latency(i, j) / 1000000);
		}
	}

	for (i = 0; i < conf.ncuda + conf.nopencl + 1; i++)
	{
		char i_name[16];
		bus_name(&conf, i_name, sizeof(i_name), i);

		for (j = 0; j < conf.ncuda + conf.nopencl + 1; j++)
		{
			char j_name[16];
			if (j == i)
				continue;
			bus_name(&conf, j_name, sizeof(j_name), j);
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
	MSG_process_create("main", &starpu_main, NULL, xbt_dynar_get_as(hosts, 0, msg_host_t));
	xbt_dynar_free(&hosts);

	MSG_main();
	return 0;
}

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
		STARPU_ASSERT_MSG(!_STARPU_IS_ZERO(length) && !isnan(length),
			"Codelet %s does not have a perfmodel, or is not calibrated enough",
			_starpu_job_get_model_name(j));
	}

	simgrid_task = MSG_task_create(_starpu_job_get_model_name(j),
			length/1000000.0*MSG_get_host_speed(MSG_host_self()),
			0, NULL);
	MSG_task_execute(simgrid_task);
}

msg_task_t _starpu_simgrid_transfer_task_create(unsigned src_node, unsigned dst_node, size_t size)
{
	msg_host_t *hosts = calloc(2, sizeof(*hosts));
	double *computation = calloc(2, sizeof(*computation));
	double *communication = calloc(4, sizeof(*communication));

	hosts[0] = _starpu_simgrid_memory_node_get_host(src_node);
	hosts[1] = _starpu_simgrid_memory_node_get_host(dst_node);
	communication[1] = size;

	return MSG_parallel_task_create("copy", 2, hosts, computation, communication, NULL);
}

struct completion {
	msg_task_t task;
	unsigned *finished;
	_starpu_pthread_mutex_t *mutex;
	_starpu_pthread_cond_t *cond;
};

int transfer_execute(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	struct completion *completion = MSG_process_get_data(MSG_process_self());
	MSG_task_execute(completion->task);
	MSG_task_destroy(completion->task);
	_STARPU_PTHREAD_MUTEX_LOCK(completion->mutex);
	*completion->finished = 1;
	_STARPU_PTHREAD_COND_BROADCAST(completion->cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(completion->mutex);
	free(completion);
	return 0;
}

void _starpu_simgrid_post_task(msg_task_t task, unsigned *finished, _starpu_pthread_mutex_t *mutex, _starpu_pthread_cond_t *cond)
{
	struct completion *completion = malloc(sizeof (*completion));
	completion->task = task;
	completion->finished = finished;
	completion->mutex = mutex;
	completion->cond = cond;
	xbt_dynar_t hosts = MSG_hosts_as_dynar();
	MSG_process_create("transfer task", transfer_execute, completion, xbt_dynar_get_as(hosts, 0, msg_host_t));
	xbt_dynar_free(&hosts);
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
