/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
#if defined(HAVE_SIMGRID_SIMDAG_H) && (SIMGRID_VERSION >= 31300)
#include <simgrid/simdag.h>
#endif

#ifdef STARPU_SIMGRID
#ifdef HAVE_GETRLIMIT
#include <sys/resource.h>
#endif
#include <simgrid/simix.h>
#ifdef STARPU_HAVE_SIMGRID_HOST_H
#include <simgrid/host.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_ENGINE_H
#include <simgrid/engine.h>
#endif
#ifdef STARPU_HAVE_XBT_CONFIG_H
#include <xbt/config.h>
#endif
#include <smpi/smpi.h>

#pragma weak starpu_main
extern int starpu_main(int argc, char *argv[]);
#if SIMGRID_VERSION < 31600
#pragma weak smpi_main
extern int smpi_main(int (*realmain) (int argc, char *argv[]), int argc, char *argv[]);
#endif
#pragma weak _starpu_mpi_simgrid_init
extern int _starpu_mpi_simgrid_init(int argc, char *argv[]);

#pragma weak smpi_process_set_user_data
#if !HAVE_DECL_SMPI_PROCESS_SET_USER_DATA && !defined(smpi_process_set_user_data)
extern void smpi_process_set_user_data(void *);
#endif

/* 1 when MSG_init was done, 2 when initialized through redirected main, 3 when
 * initialized through MSG_process_attach */
static int simgrid_started;

static int simgrid_transfer_cost = 1;

static int runners_running;
starpu_pthread_queue_t _starpu_simgrid_transfer_queue[STARPU_MAXNODES];
static struct transfer_runner
{
	struct transfer *first_transfer, *last_transfer;
	starpu_sem_t sem;
	starpu_pthread_t runner;
} transfer_runner[STARPU_MAXNODES][STARPU_MAXNODES];
static void *transfer_execute(void *arg);

starpu_pthread_queue_t _starpu_simgrid_task_queue[STARPU_NMAXWORKERS];
static struct worker_runner
{
	struct task *first_task, *last_task;
	starpu_sem_t sem;
	starpu_pthread_t runner;
} worker_runner[STARPU_NMAXWORKERS];
static void *task_execute(void *arg);

#ifdef HAVE_SG_ACTOR_ON_EXIT
static void on_exit_backtrace(int failed, void *data STARPU_ATTRIBUTE_UNUSED)
{
	if (failed)
		xbt_backtrace_display_current();
}
#endif

void _starpu_simgrid_actor_setup(void)
{
#ifdef HAVE_SG_ACTOR_ON_EXIT
	sg_actor_on_exit(on_exit_backtrace, NULL);
#endif
}

#if defined(HAVE_SG_ZONE_GET_BY_NAME) || defined(sg_zone_get_by_name)
#define HAVE_STARPU_SIMGRID_GET_AS_BY_NAME
sg_netzone_t _starpu_simgrid_get_as_by_name(const char *name)
{
	return sg_zone_get_by_name(name);
}
#elif defined(HAVE_MSG_ZONE_GET_BY_NAME) || defined(MSG_zone_get_by_name)
#define HAVE_STARPU_SIMGRID_GET_AS_BY_NAME
msg_as_t _starpu_simgrid_get_as_by_name(const char *name)
{
	return MSG_zone_get_by_name(name);
}
#elif defined(HAVE_MSG_GET_AS_BY_NAME) || defined(MSG_get_as_by_name)
#define HAVE_STARPU_SIMGRID_GET_AS_BY_NAME
msg_as_t _starpu_simgrid_get_as_by_name(const char *name)
{
	return MSG_get_as_by_name(name);
}
#elif defined(HAVE_MSG_ENVIRONMENT_GET_ROUTING_ROOT) || defined(MSG_environment_as_get_routing_sons)
#define HAVE_STARPU_SIMGRID_GET_AS_BY_NAME
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
#endif /* HAVE_MSG_ENVIRONMENT_GET_ROUTING_ROOT */

int _starpu_simgrid_get_nbhosts(const char *prefix)
{
	int ret;
#ifdef HAVE_SG_HOST_LIST
	sg_host_t *hosts_list = NULL;
#endif
	xbt_dynar_t hosts = NULL;
	unsigned i, nb = 0;
	unsigned len = strlen(prefix);

	if (_starpu_simgrid_running_smpi())
	{
#ifdef HAVE_STARPU_SIMGRID_GET_AS_BY_NAME
		char new_prefix[32];
		char name[32];
		STARPU_ASSERT(starpu_mpi_world_rank);
		snprintf(name, sizeof(name), STARPU_MPI_AS_PREFIX"%d", starpu_mpi_world_rank());
#if defined(HAVE_MSG_ZONE_GET_HOSTS) || defined(HAVE_SG_ZONE_GET_HOSTS) || defined(MSG_zone_get_hosts) || defined(sg_zone_get_hosts)
		hosts = xbt_dynar_new(sizeof(sg_host_t), NULL);
#  if defined(HAVE_SG_ZONE_GET_HOSTS) || defined(sg_zone_get_hosts)
		sg_zone_get_hosts(_starpu_simgrid_get_as_by_name(name), hosts);
#  else
		MSG_zone_get_hosts(_starpu_simgrid_get_as_by_name(name), hosts);
#  endif
#else
		hosts = MSG_environment_as_get_hosts(_starpu_simgrid_get_as_by_name(name));
#endif
		snprintf(new_prefix, sizeof(new_prefix), "%s-%s", name, prefix);
		prefix = new_prefix;
		len = strlen(prefix);
#else
		STARPU_ABORT_MSG("can not continue without an implementation for _starpu_simgrid_get_as_by_name");
#endif /* HAVE_STARPU_SIMGRID_GET_AS_BY_NAME */
	}
	else
	{
#ifdef HAVE_SG_HOST_LIST
		hosts_list = sg_host_list();
		nb = sg_host_count();
#elif defined(STARPU_HAVE_SIMGRID_HOST_H)
		hosts = sg_hosts_as_dynar();
#else
		hosts = MSG_hosts_as_dynar();
#endif
	}
	if (hosts)
		nb = xbt_dynar_length(hosts);

	ret = 0;
	for (i = 0; i < nb; i++)
	{
		const char *name;
#ifdef HAVE_SG_HOST_LIST
		if (hosts_list)
			name = sg_host_get_name(hosts_list[i]);
		else
#endif
#if defined(STARPU_HAVE_SIMGRID_HOST_H)
			name = sg_host_get_name(xbt_dynar_get_as(hosts, i, sg_host_t));
#else
			name = MSG_host_get_name(xbt_dynar_get_as(hosts, i, msg_host_t));
#endif
		if (!strncmp(name, prefix, len))
			ret++;
	}
	if (hosts)
		xbt_dynar_free(&hosts);
	return ret;
}

unsigned long long _starpu_simgrid_get_memsize(const char *prefix, unsigned devid)
{
	char name[32];
	starpu_sg_host_t host;
	const char *memsize;

	snprintf(name, sizeof(name), "%s%u", prefix, devid);

	host = _starpu_simgrid_get_host_by_name(name);
	if (!host)
		return 0;

#ifdef HAVE_SG_HOST_GET_PROPERTIES
	if (!sg_host_get_properties(host))
#else
	if (!MSG_host_get_properties(host))
#endif
		return 0;

#ifdef HAVE_SG_HOST_GET_PROPERTIES
	memsize = sg_host_get_property_value(host, "memsize");
#else
	memsize = MSG_host_get_property_value(host, "memsize");
#endif
	if (!memsize)
		return 0;

	return atoll(memsize);
}

starpu_sg_host_t _starpu_simgrid_get_host_by_name(const char *name)
{
	if (_starpu_simgrid_running_smpi())
	{
		char mpiname[32];
		STARPU_ASSERT(starpu_mpi_world_rank);
		snprintf(mpiname, sizeof(mpiname), STARPU_MPI_AS_PREFIX"%d-%s", starpu_mpi_world_rank(), name);
#ifdef STARPU_HAVE_SIMGRID_HOST_H
		return sg_host_by_name(mpiname);
#else
		return MSG_get_host_by_name(mpiname);
#endif
	}
	else
#ifdef STARPU_HAVE_SIMGRID_HOST_H
		return sg_host_by_name(name);
#else
		return MSG_get_host_by_name(name);
#endif
}

starpu_sg_host_t _starpu_simgrid_get_host_by_worker(struct _starpu_worker *worker)
{
	char *prefix;
	char name[16];
	starpu_sg_host_t host;
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
	snprintf(name, sizeof(name), "%s%u", prefix, worker->devid);
	host =  _starpu_simgrid_get_host_by_name(name);
	STARPU_ASSERT_MSG(host, "Could not find host %s!", name);
	return host;
}

/* Simgrid up to 3.15 would rename main into smpi_simulated_main_, and call that
 * from SMPI initialization
 * In case the MPI application didn't use smpicc to build the file containing
 * main(), but included our #define main starpu_main, try to cope by calling
 * starpu_main */
int _starpu_smpi_simulated_main_(int argc, char *argv[])
{
	if (!starpu_main)
	{
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h or starpu_simgrid_wrap.h included, to properly rename it into starpu_main\n");
	}

	return starpu_main(argc, argv);
}
int smpi_simulated_main_(int argc, char *argv[]) __attribute__((weak, alias("_starpu_smpi_simulated_main_")));

/* This is used to start a non-MPI simgrid environment */
void _starpu_start_simgrid(int *argc, char **argv)
{
	char path[256];

	if (simgrid_started)
		return;

	simgrid_started = 1;

#if defined(STARPU_SIMGRID_HAVE_SIMGRID_INIT) && defined(HAVE_SG_ACTOR_INIT)
	simgrid_init(argc, argv);
#else
	MSG_init(argc, argv);
#endif
	/* Simgrid uses tiny stacks by default.  This comes unexpected to our users.  */
	unsigned stack_size = 8192;
#ifdef HAVE_GETRLIMIT
	struct rlimit rlim;
	if (getrlimit(RLIMIT_STACK, &rlim) == 0 && rlim.rlim_cur != 0 && rlim.rlim_cur != RLIM_INFINITY)
		stack_size = rlim.rlim_cur / 1024;
#endif

#ifdef HAVE_SG_CFG_SET_INT
	sg_cfg_set_int("contexts/stack-size", stack_size);
#elif SIMGRID_VERSION < 31300
	extern xbt_cfg_t _sg_cfg_set;
	xbt_cfg_set_int(_sg_cfg_set, "contexts/stack_size", stack_size);
#else
	xbt_cfg_set_int("contexts/stack-size", stack_size);
#endif

	/* Load XML platform */
#if SIMGRID_VERSION < 31300
	_starpu_simgrid_get_platform_path(3, path, sizeof(path));
#else
	_starpu_simgrid_get_platform_path(4, path, sizeof(path));
#endif
#if defined(STARPU_SIMGRID_HAVE_SIMGRID_INIT) && defined(HAVE_SG_ACTOR_INIT)
	simgrid_load_platform(path);
#else
	MSG_create_environment(path);
#endif
	int limit_bandwidth = starpu_get_env_number("STARPU_LIMIT_BANDWIDTH");
	if (limit_bandwidth >= 0)
	{
#ifdef HAVE_SG_LINK_BANDWIDTH_SET
		sg_link_t *links = sg_link_list();
		int count = sg_link_count(), i;
		for (i = 0; i < count; i++) {
			sg_link_bandwidth_set(links[i], limit_bandwidth * 1000000.);
		}
#else
		_STARPU_DISP("Warning: STARPU_LIMIT_BANDWIDTH set to %d but this requires simgrid 3.26, thus ignored\n", limit_bandwidth);
#endif
	}

	simgrid_transfer_cost = starpu_get_env_number_default("STARPU_SIMGRID_TRANSFER_COST", 1);
}

static int main_ret;

int do_starpu_main(int argc, char *argv[])
{
	/* FIXME: Ugly work-around for bug in simgrid: the MPI context is not properly set at MSG process startup */
	starpu_sleep(0.000001);
	_starpu_simgrid_actor_setup();

	if (!starpu_main)
	{
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h or starpu_simgrid_wrap.h included, to properly rename it into starpu_main\n");
	}

	main_ret = starpu_main(argc, argv);
	return main_ret;
}

/* We need it only when using smpi */
#pragma weak smpi_process_get_user_data
extern void *smpi_process_get_user_data();

/* This is hopefully called before the application and simgrid */
#undef main
#pragma weak main
int main(int argc, char **argv)
{
#ifdef HAVE_SG_CONFIG_CONTINUE_AFTER_HELP
	sg_config_continue_after_help();
#endif
	if (_starpu_simgrid_running_smpi())
	{
		if (!smpi_process_get_user_data)
		{
			_STARPU_ERROR("Your version of simgrid does not provide smpi_process_get_user_data, we can not continue without it\n");
		}

#if SIMGRID_VERSION >= 31600
		/* Recent versions of simgrid dlopen() us, so we don't need to
		 * do circumvolutions, just init MPI early and run the application's main */
		return _starpu_mpi_simgrid_init(argc, argv);
#else
		/* Oops, we are running old SMPI, let it start Simgrid, and we'll
		 * take back hand in _starpu_simgrid_init from starpu_init() */
		return smpi_main(_starpu_mpi_simgrid_init, argc, argv);
#endif
	}

        /* Already initialized?  It probably has been done through a
         * constructor and MSG_process_attach, directly jump to real main */
	if (simgrid_started == 3)
	{
		return do_starpu_main(argc, argv);
	}

	/* Managed to catch application's main, initialize simgrid first */
	_starpu_start_simgrid(&argc, argv);

	simgrid_started = 2;

	/* Create a simgrid process for main */
	char **argv_cpy;
	_STARPU_MALLOC(argv_cpy, argc * sizeof(char*));
	int i;
	for (i = 0; i < argc; i++)
		argv_cpy[i] = strdup(argv[i]);

	/* Run the application in a separate thread */
	_starpu_simgrid_actor_create("main", &do_starpu_main, _starpu_simgrid_get_host_by_name("MAIN"), argc, argv_cpy);

	/* And run maestro in the main thread */
#if defined(STARPU_SIMGRID_HAVE_SIMGRID_INIT) && defined(HAVE_SG_ACTOR_INIT)
	simgrid_run();
#else
	MSG_main();
#endif
	return main_ret;
}

#if defined(HAVE_MSG_PROCESS_ATTACH) || defined(MSG_process_attach) || defined(HAVE_SG_ACTOR_ATTACH)
static void maestro(void *data STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_SIMGRID_HAVE_SIMGRID_INIT) && defined(HAVE_SG_ACTOR_INIT)
	simgrid_run();
#else
	MSG_main();
#endif
}
#endif

/* This is called early from starpu_init, so thread functions etc. can work */
void _starpu_simgrid_init_early(int *argc STARPU_ATTRIBUTE_UNUSED, char ***argv STARPU_ATTRIBUTE_UNUSED)
{
#ifdef HAVE_SG_CONFIG_CONTINUE_AFTER_HELP
	sg_config_continue_after_help();
#endif
#if defined(HAVE_MSG_PROCESS_ATTACH) || defined(MSG_process_attach) || defined(HAVE_SG_ACTOR_ATTACH)
	if (simgrid_started < 2 && !_starpu_simgrid_running_smpi())
	{
		/* "Cannot create_maestro with this ContextFactory.
		 * Try using --cfg=contexts/factory:thread instead."
		 * See https://github.com/simgrid/simgrid/issues/141 */
		_STARPU_DISP("Warning: In simgrid mode, the file containing the main() function of this application should to be compiled with starpu.h or starpu_simgrid_wrap.h included, to properly rename it into starpu_main to avoid having to use --cfg=contexts/factory:thread which reduces performance\n");
#if SIMGRID_VERSION >= 31400 /* Only recent versions of simgrid support setting sg_cfg_set_string before starting simgrid */
#  ifdef HAVE_SG_CFG_SET_INT
		sg_cfg_set_string("contexts/factory", "thread");
#  else
		xbt_cfg_set_string("contexts/factory", "thread");
#  endif
#endif
		/* We didn't catch application's main. */
		/* Start maestro as a separate thread */
		SIMIX_set_maestro(maestro, NULL);
		/* Initialize simgrid */
		_starpu_start_simgrid(argc, *argv);

		/* And attach the main thread to the main simgrid process */
		void **tsd;
		_STARPU_CALLOC(tsd, MAX_TSD+1, sizeof(void*));

#if defined(HAVE_SG_ACTOR_ATTACH) && defined (HAVE_SG_ACTOR_DATA)
		sg_actor_t actor = sg_actor_attach("main", NULL, _starpu_simgrid_get_host_by_name("MAIN"), NULL);
		sg_actor_data_set(actor, tsd);
#else
		MSG_process_attach("main", tsd, _starpu_simgrid_get_host_by_name("MAIN"), NULL);
#endif

		/* We initialized through MSG_process_attach */
		simgrid_started = 3;
	}
#endif

	if (!simgrid_started && !starpu_main && !_starpu_simgrid_running_smpi())
	{
                /* Oops, we don't have MSG_process_attach and didn't catch the
                 * 'main' symbol, there is no way for us */
		_STARPU_ERROR("In simgrid mode, the file containing the main() function of this application needs to be compiled with starpu.h or starpu_simgrid_wrap.h included, to properly rename it into starpu_main\n");
	}
	if (_starpu_simgrid_running_smpi())
	{
#ifndef STARPU_STATIC_ONLY
		_STARPU_ERROR("Simgrid currently does not support privatization for dynamically-linked libraries in SMPI. Please reconfigure and build StarPU with --disable-shared");
#endif
#if defined(HAVE_MSG_PROCESS_USERDATA_INIT) && !defined(HAVE_SG_ACTOR_DATA)
		MSG_process_userdata_init();
#endif
		void **tsd;
		_STARPU_CALLOC(tsd, MAX_TSD+1, sizeof(void*));
#ifdef HAVE_SG_ACTOR_DATA
		sg_actor_data_set(sg_actor_self(), tsd);
#else
		smpi_process_set_user_data(tsd);
#endif
	}
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
		starpu_pthread_queue_init(&_starpu_simgrid_transfer_queue[i]);
	for (i = 0; i < STARPU_NMAXWORKERS; i++)
		starpu_pthread_queue_init(&_starpu_simgrid_task_queue[i]);
}

/* This is called late from starpu_init, to start task executors */
void _starpu_simgrid_init(void)
{
	unsigned i;
	runners_running = 1;
	for (i = 0; i < starpu_worker_get_count(); i++)
	{
		char s[32];
		snprintf(s, sizeof(s), "worker %u runner", i);
		starpu_sem_init(&worker_runner[i].sem, 0, 0);
		starpu_pthread_create_on(s, &worker_runner[i].runner, NULL, task_execute, (void*)(uintptr_t) i, _starpu_simgrid_get_host_by_worker(_starpu_get_worker_struct(i)));
	}
}

void _starpu_simgrid_deinit_late(void)
{
#if defined(HAVE_MSG_PROCESS_ATTACH) || defined(MSG_process_attach) || defined(HAVE_SG_ACTOR_ATTACH)
	if (simgrid_started == 3)
	{
		/* Started with MSG_process_attach, now detach */
#ifdef HAVE_SG_ACTOR_ATTACH
		sg_actor_detach();
#else
		MSG_process_detach();
#endif
		simgrid_started = 0;
	}
#endif
}

void _starpu_simgrid_deinit(void)
{
	unsigned i, j;
	runners_running = 0;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		for (j = 0; j < STARPU_MAXNODES; j++)
		{
			struct transfer_runner *t = &transfer_runner[i][j];
			if (t->runner)
			{
				starpu_sem_post(&t->sem);
#ifdef STARPU_HAVE_SIMGRID_ACTOR_H
				sg_actor_join(t->runner, 1000000);
#elif SIMGRID_VERSION >= 31400
				MSG_process_join(t->runner, 1000000);
#else
				starpu_sleep(1);
#endif
				STARPU_ASSERT(t->first_transfer == NULL);
				STARPU_ASSERT(t->last_transfer == NULL);
				starpu_sem_destroy(&t->sem);
			}
		}
		/* FIXME: queue not empty at this point, needs proper unregistration */
		/* starpu_pthread_queue_destroy(&_starpu_simgrid_transfer_queue[i]); */
	}
	for (i = 0; i < starpu_worker_get_count(); i++)
	{
		struct worker_runner *w = &worker_runner[i];
		starpu_sem_post(&w->sem);
#ifdef STARPU_HAVE_SIMGRID_ACTOR_H
		sg_actor_join(w->runner, 1000000);
#elif SIMGRID_VERSION >= 31400
		MSG_process_join(w->runner, 1000000);
#else
		starpu_sleep(1);
#endif
		STARPU_ASSERT(w->first_task == NULL);
		STARPU_ASSERT(w->last_task == NULL);
		starpu_sem_destroy(&w->sem);
		starpu_pthread_queue_destroy(&_starpu_simgrid_task_queue[i]);
	}

#if SIMGRID_VERSION >= 31300
	/* clean-atexit introduced in simgrid 3.13 */
#  ifdef HAVE_SG_CFG_SET_INT
	if ( sg_cfg_get_boolean("debug/clean-atexit"))
#  elif SIMGRID_VERSION >= 32300
	if ( xbt_cfg_get_boolean("debug/clean-atexit"))
#  else
	if ( xbt_cfg_get_boolean("clean-atexit"))
#  endif
	{
		_starpu_simgrid_deinit_late();
	}
#endif
}

/*
 * Tasks
 */

struct task
{
#ifdef HAVE_SG_ACTOR_SELF_EXECUTE
	double flops;
#else
	msg_task_t task;
#endif

	/* communication termination signalization */
	unsigned *finished;

	/* Next task on this worker */
	struct task *next;
};

/* Actually execute the task.  */
static void *task_execute(void *arg)
{
	unsigned workerid = (uintptr_t) arg;
	struct worker_runner *w = &worker_runner[workerid];

	_STARPU_DEBUG("worker runner %u started\n", workerid);
	while (1)
	{
		struct task *task;

		starpu_sem_wait(&w->sem);
		if (!runners_running)
			break;

		task = w->first_task;
		w->first_task = task->next;
		if (w->last_task == task)
			w->last_task = NULL;

		_STARPU_DEBUG("task %p started\n", task);
#ifdef HAVE_SG_ACTOR_EXECUTE
		sg_actor_execute(task->flops);
#elif defined(HAVE_SG_ACTOR_SELF_EXECUTE)
		sg_actor_self_execute(task->flops);
#else
		MSG_task_execute(task->task);
		MSG_task_destroy(task->task);
#endif
		_STARPU_DEBUG("task %p finished\n", task);

		*task->finished = 1;
		/* The worker which started this task may be sleeping out of tasks, wake it  */
		_starpu_wake_worker_relax(workerid);

		free(task);
	}
	_STARPU_DEBUG("worker %u stopped\n", workerid);
	return 0;
}

/* Wait for completion of all asynchronous tasks for this worker */
void _starpu_simgrid_wait_tasks(int workerid)
{
	struct task *task = worker_runner[workerid].last_task;
	if (!task)
		return;

	unsigned *finished = task->finished;
	starpu_pthread_wait_t wait;
	starpu_pthread_wait_init(&wait);
	starpu_pthread_queue_register(&wait, &_starpu_simgrid_task_queue[workerid]);

	while(1)
	{
		starpu_pthread_wait_reset(&wait);
		if (*finished)
			break;
		starpu_pthread_wait_wait(&wait);
	}
	starpu_pthread_queue_unregister(&wait, &_starpu_simgrid_task_queue[workerid]);
	starpu_pthread_wait_destroy(&wait);
}

/* Task execution submitted by StarPU */
void _starpu_simgrid_submit_job(int workerid, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch, double length, unsigned *finished)
{
	struct starpu_task *starpu_task = j->task;
	double flops;
#ifndef HAVE_SG_ACTOR_SELF_EXECUTE
	msg_task_t simgrid_task;
#endif

	if (j->internal)
		/* This is not useful to include in simulation (and probably
		 * doesn't have a perfmodel anyway) */
		return;

	if (isnan(length))
	{
		length = starpu_task_expected_length(starpu_task, perf_arch, j->nimpl);
		STARPU_ASSERT_MSG(!_STARPU_IS_ZERO(length) && !isnan(length),
				  "Codelet %s does not have a perfmodel (in directory %s), or is not calibrated enough, please re-run in non-simgrid mode until it is calibrated, or fix the STARPU_HOSTNAME and STARPU_PERF_MODEL_DIR environment variables",
				  _starpu_job_get_model_name(j), _starpu_get_perf_model_dir_codelet());
                /* TODO: option to add variance according to performance model,
                 * to be able to easily check scheduling robustness */
	}

#if defined(HAVE_SG_HOST_SPEED) || defined(sg_host_speed)
#  if defined(HAVE_SG_HOST_SELF) || defined(sg_host_self)
	flops = length/1000000.0*sg_host_speed(sg_host_self());
#  else
	flops = length/1000000.0*sg_host_speed(MSG_host_self());
#  endif
#elif defined HAVE_MSG_HOST_GET_SPEED || defined(MSG_host_get_speed)
	flops = length/1000000.0*MSG_host_get_speed(MSG_host_self());
#else
	flops = length/1000000.0*MSG_get_host_speed(MSG_host_self());
#endif

#ifndef HAVE_SG_ACTOR_SELF_EXECUTE
	simgrid_task = MSG_task_create(_starpu_job_get_task_name(j), flops, 0, NULL);
#endif

	if (finished == NULL)
	{
		/* Synchronous execution */
		/* First wait for previous tasks */
		_starpu_simgrid_wait_tasks(workerid);
#ifdef HAVE_SG_ACTOR_EXECUTE
		sg_actor_execute(flops);
#elif defined(HAVE_SG_ACTOR_SELF_EXECUTE)
		sg_actor_self_execute(flops);
#else
		MSG_task_execute(simgrid_task);
		MSG_task_destroy(simgrid_task);
#endif
	}
	else
	{
		/* Asynchronous execution */
		struct task *task;
		struct worker_runner *w = &worker_runner[workerid];
		_STARPU_MALLOC(task, sizeof(*task));
#ifdef HAVE_SG_ACTOR_SELF_EXECUTE
		task->flops = flops;
#else
		task->task = simgrid_task;
#endif
		task->finished = finished;
		*finished = 0;
		task->next = NULL;
		/* Sleep 10µs for the GPU task queueing */
		if (_starpu_simgrid_queue_malloc_cost())
			starpu_sleep(0.000010);
		if (w->last_task)
		{
			/* Already running a task, queue */
			w->last_task->next = task;
			w->last_task = task;
		}
		else
		{
			STARPU_ASSERT(!w->first_task);
			w->first_task = task;
			w->last_task = task;
		}
		starpu_sem_post(&w->sem);
	}
}

/*
 * Transfers
 */

/* Note: simgrid is not parallel, so there is no need to hold locks for management of transfers.  */
LIST_TYPE(transfer,
#if defined(HAVE_SG_HOST_SEND_TO) || defined(HAVE_SG_HOST_SENDTO)
	size_t size;
#else
	msg_task_t task;
#endif
	int src_node;
	int dst_node;
	int run_node;

	/* communication termination signalization */
	unsigned *finished;

	/* transfers which wait for this transfer */
	struct transfer **wake;
	unsigned nwake;

	/* Number of transfers that this transfer waits for */
	unsigned nwait;

	/* Next transfer on this stream */
	struct transfer *next;
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
			&& (old_transfer->dst_node == new_transfer->src_node
			 || old_transfer->dst_node == new_transfer->dst_node))
		return 1;
	if (old_is_gpu_gpu
			&& (new_transfer->dst_node == old_transfer->src_node
			 || new_transfer->dst_node == old_transfer->dst_node))
		return 1;

	/* StarPU's constraint on CUDA transfers is using one stream per
	 * source/destination pair, which is already handled above */

	return 0;
}

static void transfer_queue(struct transfer *transfer)
{
	unsigned src = transfer->src_node;
	unsigned dst = transfer->dst_node;
	struct transfer_runner *t = &transfer_runner[src][dst];

	if (!t->runner)
	{
		/* No runner yet, start it */
		static starpu_pthread_mutex_t mutex; /* process_create may yield */
		STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		if (!t->runner)
		{
			char s[64];
			snprintf(s, sizeof(s), "transfer %u-%u runner", src, dst);
			starpu_pthread_create_on(s, &t->runner, NULL, transfer_execute, (void*)(uintptr_t)((src<<16) + dst), _starpu_simgrid_get_memnode_host(src));
			starpu_sem_init(&t->sem, 0, 0);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	}

	if (t->last_transfer)
	{
		/* Already running a transfer, queue */
		t->last_transfer->next = transfer;
		t->last_transfer = transfer;
	}
	else
	{
		STARPU_ASSERT(!t->first_transfer);
		t->first_transfer = transfer;
		t->last_transfer = transfer;
	}
	starpu_sem_post(&t->sem);
}

/* Actually execute the transfer, and then start transfers waiting for this one.  */
static void *transfer_execute(void *arg)
{
	unsigned src_dst = (uintptr_t) arg;
	unsigned src = src_dst >> 16;
	unsigned dst = src_dst & 0xffff;
	struct transfer_runner *t = &transfer_runner[src][dst];

	_STARPU_DEBUG("transfer runner %u-%u started\n", src, dst);
	while (1)
	{
		struct transfer *transfer;

		starpu_sem_wait(&t->sem);
		if (!runners_running)
			break;
		transfer = t->first_transfer;
		t->first_transfer = transfer->next;
		if (t->last_transfer == transfer)
			t->last_transfer = NULL;

#if defined(HAVE_SG_HOST_SEND_TO) || defined(HAVE_SG_HOST_SENDTO)
		if (transfer->size)
#else
		if (transfer->task)
#endif
		{
			_STARPU_DEBUG("transfer %p started\n", transfer);
#if defined(HAVE_SG_HOST_SEND_TO) || defined(HAVE_SG_HOST_SENDTO)
#ifdef HAVE_SG_HOST_SENDTO
			sg_host_sendto
#else
			sg_host_send_to
#endif
				(_starpu_simgrid_memory_node_get_host(transfer->src_node),
					_starpu_simgrid_memory_node_get_host(transfer->dst_node),
					transfer->size);
#else
			MSG_task_execute(transfer->task);
			MSG_task_destroy(transfer->task);
#endif
			_STARPU_DEBUG("transfer %p finished\n", transfer);
		}

		*transfer->finished = 1;
		transfer_list_erase(&pending, transfer);

		/* The workers which started this request may be sleeping out of tasks, wake it  */
		_starpu_wake_all_blocked_workers_on_node(transfer->run_node);

		unsigned i;
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
				transfer_queue(wake);
			}
		}
		free(transfer->wake);
		free(transfer);
	}

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
			_STARPU_REALLOC(old->wake, (old->nwake + 1) * sizeof(old->wake));
			old->wake[old->nwake] = transfer;
			old->nwake++;
		}
	}

	transfer_list_push_front(&pending, transfer);

	if (!transfer->nwait)
	{
		_STARPU_DEBUG("transfer %p waits for nobody, starting\n", transfer);
		transfer_queue(transfer);
	}
}

int _starpu_simgrid_wait_transfer_event(union _starpu_async_channel_event *event)
{
	/* this is not associated to a request so it's synchronous */
	starpu_pthread_wait_t wait;
	starpu_pthread_wait_init(&wait);
	starpu_pthread_queue_register(&wait, event->queue);

	while(1)
	{
		starpu_pthread_wait_reset(&wait);
		if (event->finished)
			break;
		starpu_pthread_wait_wait(&wait);
	}
	starpu_pthread_queue_unregister(&wait, event->queue);
	starpu_pthread_wait_destroy(&wait);
	return 0;
}

int _starpu_simgrid_test_transfer_event(union _starpu_async_channel_event *event)
{
	return event->finished;
}

/* Wait for completion of all transfers */
static void _starpu_simgrid_wait_transfers(void)
{
	unsigned finished = 0;
	struct transfer *sync = transfer_new();
	struct transfer *cur;

#if defined(HAVE_SG_HOST_SEND_TO) || defined(HAVE_SG_HOST_SENDTO)
	sync->size = 0;
#else
	sync->task = NULL;
#endif
	sync->finished = &finished;

	sync->src_node = STARPU_MAIN_RAM;
	sync->dst_node = STARPU_MAIN_RAM;
	sync->run_node = STARPU_MAIN_RAM;

	sync->wake = NULL;
	sync->nwake = 0;
	sync->nwait = 0;
	sync->next = NULL;

	for (cur  = transfer_list_begin(&pending);
	     cur != transfer_list_end(&pending);
	     cur  = transfer_list_next(cur))
	{
		sync->nwait++;
		_STARPU_REALLOC(cur->wake, (cur->nwake + 1) * sizeof(cur->wake));
		cur->wake[cur->nwake] = sync;
		cur->nwake++;
	}

	if (sync->nwait == 0)
	{
		/* No transfer to wait for */
		free(sync);
		return;
	}

	/* Push synchronization pseudo-transfer */
	transfer_list_push_front(&pending, sync);

	/* And wait for it */
	starpu_pthread_wait_t wait;
	starpu_pthread_wait_init(&wait);
	starpu_pthread_queue_register(&wait, &_starpu_simgrid_transfer_queue[STARPU_MAIN_RAM]);
	while(1)
	{
		starpu_pthread_wait_reset(&wait);
		if (finished)
			break;
		starpu_pthread_wait_wait(&wait);
	}
	starpu_pthread_queue_unregister(&wait, &_starpu_simgrid_transfer_queue[STARPU_MAIN_RAM]);
	starpu_pthread_wait_destroy(&wait);
}

/* Data transfer issued by StarPU */
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req)
{
	/* Simgrid does not like 0-bytes transfers */
	if (!size)
		return 0;

	/* Explicitly disabled by user? */
	if (!simgrid_transfer_cost)
		return 0;

	union _starpu_async_channel_event *event, myevent;
	double start = 0.;
	struct transfer *transfer = transfer_new();

	_STARPU_DEBUG("creating transfer %p for %lu bytes\n", transfer, (unsigned long) size);

#if defined(HAVE_SG_HOST_SEND_TO) || defined(HAVE_SG_HOST_SENDTO)
	transfer->size = size;
#else
	msg_task_t task;
	starpu_sg_host_t *hosts;
	double *computation;
	double *communication;

	_STARPU_CALLOC(hosts, 2, sizeof(*hosts));
	_STARPU_CALLOC(computation, 2, sizeof(*computation));
	_STARPU_CALLOC(communication, 4, sizeof(*communication));

	hosts[0] = _starpu_simgrid_memory_node_get_host(src_node);
	hosts[1] = _starpu_simgrid_memory_node_get_host(dst_node);
	STARPU_ASSERT(hosts[0] != hosts[1]);
	communication[1] = size;

	task = MSG_parallel_task_create("copy", 2, hosts, computation, communication, NULL);

	transfer->task = task;
#endif
	transfer->src_node = src_node;
	transfer->dst_node = dst_node;
	transfer->run_node = starpu_worker_get_local_memory_node();

	if (req)
		event = &req->async_channel.event;
	else
		event = &myevent;
	event->finished = 0;
	transfer->finished = &event->finished;
	event->queue = &_starpu_simgrid_transfer_queue[transfer->run_node];

	transfer->wake = NULL;
	transfer->nwake = 0;
	transfer->nwait = 0;
	transfer->next = NULL;

	if (req)
		starpu_interface_start_driver_copy_async(src_node, dst_node, &start);

	/* Sleep 10µs for the GPU transfer queueing */
	if (_starpu_simgrid_queue_malloc_cost())
		starpu_sleep(0.000010);
	transfer_submit(transfer);
	/* Note: from here, transfer might be already freed */

	if (req)
	{
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);
		starpu_interface_data_copy(src_node, dst_node, size);
		return -EAGAIN;
	}
	else
	{
		/* this is not associated to a request so it's synchronous */
		_starpu_simgrid_wait_transfer_event(event);
		return 0;
	}
}

/* Sync all GPUs (used on CUDA Free, typically) */
void _starpu_simgrid_sync_gpus(void)
{
	_starpu_simgrid_wait_transfers();
}

int
_starpu_simgrid_thread_start(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[])
{
	void *(*f)(void*) = (void*) (uintptr_t) strtol(argv[0], NULL, 16);
	void *arg = (void*) (uintptr_t) strtol(argv[1], NULL, 16);

	/* FIXME: Ugly work-around for bug in simgrid: the MPI context is not properly set at MSG process startup */
	starpu_sleep(0.000001);
	_starpu_simgrid_actor_setup();

	/* _args is freed with process context */
	f(arg);
	return 0;
}

starpu_pthread_t _starpu_simgrid_actor_create(const char *name, xbt_main_func_t code, starpu_sg_host_t host, int argc, char *argv[])
{
	void **tsd;
	starpu_pthread_t actor;
	_STARPU_CALLOC(tsd, MAX_TSD+1, sizeof(void*));
#ifdef HAVE_SG_ACTOR_INIT
	actor = sg_actor_init(name, host);
	sg_actor_data_set(actor, tsd);
	sg_actor_start(actor, code, argc, argv);
#else
	actor = MSG_process_create_with_arguments(name, code, tsd, host, argc, argv);
#ifdef HAVE_SG_ACTOR_DATA
	sg_actor_data_set(actor, tsd);
#endif
#endif
	return actor;
}

starpu_sg_host_t _starpu_simgrid_get_memnode_host(unsigned node)
{
	const char *fmt;
	char name[16];

	switch (starpu_node_get_kind(node))
	{
		case STARPU_CPU_RAM:
			fmt = "RAM";
			break;
		case STARPU_CUDA_RAM:
			fmt = "CUDA%u";
			break;
		case STARPU_OPENCL_RAM:
			fmt = "OpenCL%u";
			break;
		case STARPU_DISK_RAM:
			fmt = "DISK%u";
			break;
		default:
			STARPU_ABORT();
			break;
	}
	snprintf(name, sizeof(name), fmt, starpu_memory_node_get_devid(node));

	return _starpu_simgrid_get_host_by_name(name);
}

void _starpu_simgrid_count_ngpus(void)
{
#if (defined(HAVE_SG_LINK_NAME) || defined sg_link_name) && (SIMGRID_VERSION >= 31300)
	unsigned src, dst;
	starpu_sg_host_t ramhost = _starpu_simgrid_get_host_by_name("RAM");

	/* For each pair of memory nodes, get the route */
	for (src = 1; src < STARPU_MAXNODES; src++)
		for (dst = 1; dst < STARPU_MAXNODES; dst++)
		{
			int busid;
			starpu_sg_host_t srchost, dsthost;
#if defined(HAVE_SG_HOST_ROUTE) || defined(sg_host_route)
			xbt_dynar_t route_dynar = xbt_dynar_new(sizeof(SD_link_t), NULL);
			SD_link_t *route;
#else
			const SD_link_t *route;
#endif
			int i, routesize;
			int through;
			unsigned src2;
			unsigned ngpus;
			const char *name;

			if (dst == src)
				continue;
			busid = starpu_bus_get_id(src, dst);
			if (busid == -1)
				continue;

			srchost = _starpu_simgrid_get_memnode_host(src);
			dsthost = _starpu_simgrid_get_memnode_host(dst);
#if defined(HAVE_SG_HOST_ROUTE)  || defined(sg_host_route)
			sg_host_route(srchost, dsthost, route_dynar);
			routesize = xbt_dynar_length(route_dynar);
			route = xbt_dynar_to_array(route_dynar);
#else
			routesize = SD_route_get_size(srchost, dsthost);
			route = SD_route_get_list(srchost, dsthost);
#endif

			/* If it goes through "Host", do not care, there is no
			 * direct transfer support */
			for (i = 0; i < routesize; i++)
				if (!strcmp(sg_link_name(route[i]), "Host"))
					break;
			if (i < routesize)
				continue;

			/* Get the PCI bridge between down and up links */
			through = -1;
			for (i = 0; i < routesize; i++)
			{
				name = sg_link_name(route[i]);
				size_t len = strlen(name);
				if (!strcmp(" through", name+len-8))
					through = i;
				else if (!strcmp(" up", name+len-3))
					break;
			}
			/* Didn't find it ?! */
			if (through == -1)
			{
				_STARPU_DEBUG("Didn't find through-link for %d->%d\n", src, dst);
				continue;
			}
			name = sg_link_name(route[through]);

			/*
			 * count how many direct routes go through it between
			 * GPUs and RAM
			 */
			ngpus = 0;
			for (src2 = 1; src2 < STARPU_MAXNODES; src2++)
			{
				int numa;
				int nnumas = starpu_memory_nodes_get_numa_count();
				int found = 0;
				for (numa = 0; numa < nnumas; numa++)
					if (starpu_bus_get_id(src2, numa) != -1)
					{
						found = 1;
						break;
					}

				if (!found)
					continue;

				starpu_sg_host_t srchost2 = _starpu_simgrid_get_memnode_host(src2);
				int routesize2;
#if defined(HAVE_SG_HOST_ROUTE) || defined(sg_host_route)
				xbt_dynar_t route_dynar2 = xbt_dynar_new(sizeof(SD_link_t), NULL);
				SD_link_t *route2;
				sg_host_route(srchost2, ramhost, route_dynar2);
				routesize2 = xbt_dynar_length(route_dynar2);
				route2 = xbt_dynar_to_array(route_dynar2);
#else
				const SD_link_t *route2 = SD_route_get_list(srchost2, ramhost);
				routesize2 = SD_route_get_size(srchost2, ramhost);
#endif

				for (i = 0; i < routesize2; i++)
					if (!strcmp(name, sg_link_name(route2[i])))
					{
						/* This GPU goes through this PCI bridge to access RAM */
						ngpus++;
						break;
					}
#if defined(HAVE_SG_HOST_ROUTE) || defined(sg_host_route)
				free(route2);
#endif
			}
			_STARPU_DEBUG("%d->%d through %s, %u GPUs\n", src, dst, name, ngpus);
			starpu_bus_set_ngpus(busid, ngpus);
#if defined(HAVE_SG_HOST_ROUTE) || defined(sg_host_route)
			free(route);
#endif
		}
#endif
}

#if 0
static size_t used;

void _starpu_simgrid_data_new(size_t size)
{
	// Note: this is just declarative
	//_STARPU_DISP("data new: %zd, now %zd\n", size, used);
}

void _starpu_simgrid_data_increase(size_t size)
{
	used += size;
	_STARPU_DISP("data increase: %zd, now %zd\n", size, used);
}

void _starpu_simgrid_data_alloc(size_t size)
{
	used += size;
	_STARPU_DISP("data alloc: %zd, now %zd\n", size, used);
}

void _starpu_simgrid_data_free(size_t size)
{
	used -= size;
	_STARPU_DISP("data free: %zd, now %zd\n", size, used);
}

void _starpu_simgrid_data_transfer(size_t size, unsigned src_node, unsigned dst_node)
{
	_STARPU_DISP("data transfer %zd from %u to %u\n", size, src_node, dst_node);
}
#endif


#endif
