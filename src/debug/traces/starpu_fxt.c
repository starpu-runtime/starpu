/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2015  Université de Bordeaux
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
#include <common/config.h>
#include <common/uthash.h>
#include <string.h>

#ifdef STARPU_HAVE_POTI
#include <poti.h>
#define STARPU_POTI_STR_LEN 200
#endif

#ifdef STARPU_USE_FXT
#include "starpu_fxt.h"
#include <inttypes.h>
#include <starpu_hash.h>

#define CPUS_WORKER_COLORS_NB	8
#define CUDA_WORKER_COLORS_NB	9
#define OPENCL_WORKER_COLORS_NB 9
#define MIC_WORKER_COLORS_NB	9
#define SCC_WORKER_COLORS_NB	9
#define OTHER_WORKER_COLORS_NB	4

static char *cpus_worker_colors[CPUS_WORKER_COLORS_NB] = {"/greens9/7", "/greens9/6", "/greens9/5", "/greens9/4",  "/greens9/9", "/greens9/3",  "/greens9/2",  "/greens9/1"  };
static char *cuda_worker_colors[CUDA_WORKER_COLORS_NB] = {"/ylorrd9/9", "/ylorrd9/6", "/ylorrd9/3", "/ylorrd9/1", "/ylorrd9/8", "/ylorrd9/7", "/ylorrd9/4", "/ylorrd9/2",  "/ylorrd9/1"};
static char *opencl_worker_colors[OPENCL_WORKER_COLORS_NB] = {"/blues9/9", "/blues9/6", "/blues9/3", "/blues9/1", "/blues9/8", "/blues9/7", "/blues9/4", "/blues9/2",  "/blues9/1"};
static char *mic_worker_colors[MIC_WORKER_COLORS_NB] = {"/reds9/9", "/reds9/6", "/reds9/3", "/reds9/1", "/reds9/8", "/reds9/7", "/reds9/4", "/reds9/2",  "/reds9/1"};
static char *scc_worker_colors[SCC_WORKER_COLORS_NB] = {"/reds9/9", "/reds9/6", "/reds9/3", "/reds9/1", "/reds9/8", "/reds9/7", "/reds9/4", "/reds9/2",  "/reds9/1"};
static char *other_worker_colors[OTHER_WORKER_COLORS_NB] = {"/greys9/9", "/greys9/8", "/greys9/7", "/greys9/6"};
static char *worker_colors[STARPU_NMAXWORKERS];

static unsigned opencl_index = 0;
static unsigned cuda_index = 0;
static unsigned cpus_index = 0;
static unsigned mic_index = 0;
static unsigned scc_index = 0;
static unsigned other_index = 0;

/*
 * Paje trace file tools
 */

static FILE *out_paje_file;
static FILE *distrib_time;
static FILE *activity_file;
static FILE *anim_file;
static FILE *tasks_file;

struct data_info {
	unsigned long handle;
	unsigned long size;
	int mode;
};

struct task_info {
	UT_hash_handle hh;
	char *model_name;
	char *name;
	int exclude_from_dag;
	unsigned long job_id;
	uint64_t tag;
	int workerid;
	double submit_time;
	double start_time;
	double end_time;
	unsigned long footprint;
	char *parameters;
	unsigned int ndeps;
	unsigned long *dependencies;
	unsigned long ndata;
	struct data_info *data;
};

struct task_info *tasks_info;

static struct task_info *get_task(unsigned long job_id)
{
	struct task_info *task;

	HASH_FIND(hh, tasks_info, &job_id, sizeof(job_id), task);
	if (!task)
	{
		task = malloc(sizeof(*task));
		task->model_name = NULL;
		task->name = NULL;
		task->exclude_from_dag = 0;
		task->job_id = job_id;
		task->tag = 0;
		task->workerid = -1;
		task->submit_time = 0.;
		task->start_time = 0.;
		task->end_time = 0.;
		task->footprint = 0;
		task->parameters = NULL;
		task->ndeps = 0;
		task->dependencies = NULL;
		task->ndata = 0;
		task->data = NULL;
		HASH_ADD(hh, tasks_info, job_id, sizeof(task->job_id), task);
	}

	return task;
}

static void task_dump(unsigned long job_id)
{
	struct task_info *task = get_task(job_id);
	unsigned i;

	if (task->exclude_from_dag)
		goto out;

	if (task->name)
	{
		fprintf(tasks_file, "Name: %s\n", task->name);
		if (!task->model_name)
			fprintf(tasks_file, "Model: %s\n", task->name);
		free(task->name);
	}
	if (task->model_name)
	{
		fprintf(tasks_file, "Model: %s\n", task->model_name);
		free(task->model_name);
	}
	fprintf(tasks_file, "JobId: %lu\n", task->job_id);
	if (task->dependencies)
	{
		fprintf(tasks_file, "DependsOn:");
		for (i = 0; i < task->ndeps; i++)
			fprintf(tasks_file, " %lu", task->dependencies[i]);
		fprintf(tasks_file, "\n");
		free(task->dependencies);
	}
	fprintf(tasks_file, "Tag: %"PRIx64"\n", task->tag);
	if (task->workerid >= 0)
		fprintf(tasks_file, "WorkerId: %d\n", task->workerid);
	if (task->submit_time != 0.)
		fprintf(tasks_file, "SubmitTime: %f\n", task->submit_time);
	if (task->start_time != 0.)
		fprintf(tasks_file, "StartTime: %f\n", task->start_time);
	if (task->end_time != 0.)
		fprintf(tasks_file, "EndTime: %f\n", task->end_time);
	fprintf(tasks_file, "Footprint: %lx\n", task->footprint);
	if (task->parameters)
	{
		fprintf(tasks_file, "Parameters: %s\n", task->parameters);
		free(task->parameters);
	}
	if (task->data)
	{
		fprintf(tasks_file, "Handles:");
		for (i = 0; i < task->ndata; i++)
			fprintf(tasks_file, " %lx", task->data[i].handle);
		fprintf(tasks_file, "\n");
		fprintf(tasks_file, "Modes:");
		for (i = 0; i < task->ndata; i++)
			fprintf(tasks_file, " %s%s%s%s%s",
				(task->data[i].mode & STARPU_R)?"R":"",
				(task->data[i].mode & STARPU_W)?"W":"",
				(task->data[i].mode & STARPU_SCRATCH)?"S":"",
				(task->data[i].mode & STARPU_REDUX)?"X":"",
				(task->data[i].mode & STARPU_COMMUTE)?"C":"");
		fprintf(tasks_file, "\n");
		fprintf(tasks_file, "Sizes:");
		for (i = 0; i < task->ndata; i++)
			fprintf(tasks_file, " %lu", task->data[i].size);
		fprintf(tasks_file, "\n");
	}
	fprintf(tasks_file, "\n");

out:
	HASH_DEL(tasks_info, task);
	free(task);
}

static void set_next_other_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		return;
	worker_colors[workerid] = other_worker_colors[other_index++];
	if (other_index == OTHER_WORKER_COLORS_NB) other_index = 0;
}

static void set_next_cpu_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		return;
	worker_colors[workerid] = cpus_worker_colors[cpus_index++];
	if (cpus_index == CPUS_WORKER_COLORS_NB) cpus_index = 0;
}

static void set_next_cuda_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		return;
	worker_colors[workerid] = cuda_worker_colors[cuda_index++];
	if (cuda_index == CUDA_WORKER_COLORS_NB) cuda_index = 0;
}

static void set_next_opencl_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		return;
	worker_colors[workerid] = opencl_worker_colors[opencl_index++];
	if (opencl_index == OPENCL_WORKER_COLORS_NB) opencl_index = 0;
}

static void set_next_mic_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		return;
	worker_colors[workerid] = mic_worker_colors[mic_index++];
	if (mic_index == MIC_WORKER_COLORS_NB) mic_index = 0;
}

static void set_next_scc_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		return;
	worker_colors[workerid] = scc_worker_colors[scc_index++];
	if (scc_index == SCC_WORKER_COLORS_NB) scc_index = 0;
}

static const char *get_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		workerid = STARPU_NMAXWORKERS - 1;
	return worker_colors[workerid];
}

static unsigned get_colour_symbol_red(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = starpu_hash_crc32c_string(name, 0);
	return (unsigned)starpu_hash_crc32c_string("red", hash_symbol) % 1024;
}

static unsigned get_colour_symbol_green(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = starpu_hash_crc32c_string(name, 0);
	return (unsigned)starpu_hash_crc32c_string("green", hash_symbol) % 1024;
}

static unsigned get_colour_symbol_blue(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = starpu_hash_crc32c_string(name, 0);
	return (unsigned)starpu_hash_crc32c_string("blue", hash_symbol) % 1024;
}

static double last_codelet_start[STARPU_NMAXWORKERS];
/* _STARPU_FUT_DO_PROBE4STR records only 4 longs */
char _starpu_last_codelet_symbol[STARPU_NMAXWORKERS][4*sizeof(unsigned long)];
static int last_codelet_parameter[STARPU_NMAXWORKERS];
#define MAX_PARAMETERS 8
static char last_codelet_parameter_description[STARPU_NMAXWORKERS][MAX_PARAMETERS][FXT_MAX_PARAMS*sizeof(unsigned long)];

/* If more than a period of time has elapsed, we flush the profiling info,
 * otherwise they are accumulated everytime there is a new relevant event. */
#define ACTIVITY_PERIOD	75.0
static double last_activity_flush_timestamp[STARPU_NMAXWORKERS];
static double accumulated_sleep_time[STARPU_NMAXWORKERS];
static double accumulated_exec_time[STARPU_NMAXWORKERS];
static double reclaiming[STARPU_MAXNODES];

static unsigned steal_number = 0;

LIST_TYPE(_starpu_symbol_name,
	char *name;
)

static struct _starpu_symbol_name_list symbol_list;

LIST_TYPE(_starpu_communication,
	unsigned comid;
	double comm_start;
	double bandwidth;
	unsigned src_node;
	unsigned dst_node;
)

static struct _starpu_communication_list communication_list;

/*
 * Generic tools
 */

static double get_event_time_stamp(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	return (((double)(ev->time-options->file_offset))/1000000.0);
}

static int nworkers = 0;

struct worker_entry
{
	UT_hash_handle hh;
	unsigned long tid;
	int workerid;
	int sync;
} *worker_ids;

static int register_worker_id(unsigned long tid, int workerid, int sync)
{
	nworkers++;
	struct worker_entry *entry;

	HASH_FIND(hh, worker_ids, &tid, sizeof(tid), entry);

	STARPU_ASSERT_MSG(workerid < STARPU_NMAXWORKERS, "Too many workers in this trace, please increase in ./configure invocation the maximum number of CPUs and GPUs to the same value as was used for execution");

	/* only register a thread once */
	if (entry)
		return 0;

	entry = malloc(sizeof(*entry));
	entry->tid = tid;
	entry->workerid = workerid;
	entry->sync = sync;

	HASH_ADD(hh, worker_ids, tid, sizeof(tid), entry);
	return 1;
}

static int find_worker_id(unsigned long tid)
{
	struct worker_entry *entry;

	HASH_FIND(hh, worker_ids, &tid, sizeof(tid), entry);
	if (!entry)
		return -1;

	return entry->workerid;
}

static int find_sync(unsigned long tid)
{
	struct worker_entry *entry;

	HASH_FIND(hh, worker_ids, &tid, sizeof(tid), entry);
	if (!entry)
		return 0;

	return entry->sync;
}

static void update_accumulated_time(int worker, double sleep_time, double exec_time, double current_timestamp, int forceflush)
{
	accumulated_sleep_time[worker] += sleep_time;
	accumulated_exec_time[worker] += exec_time;

	/* If sufficient time has elapsed since the last flush, we have a new
	 * point in our graph */
	double elapsed = current_timestamp - last_activity_flush_timestamp[worker];
	if (forceflush || (elapsed > ACTIVITY_PERIOD))
	{
		if (activity_file)
			fprintf(activity_file, "%d\t%.9f\t%.9f\t%.9f\t%.9f\n", worker, current_timestamp, elapsed, accumulated_exec_time[worker], accumulated_sleep_time[worker]);

		/* reset the accumulated times */
		last_activity_flush_timestamp[worker] = current_timestamp;
		accumulated_sleep_time[worker] = 0.0;
		accumulated_exec_time[worker] = 0.0;
	}
}

/*
 *      Auxiliary functions for poti handling names
 */
#ifdef STARPU_HAVE_POTI
static char *memnode_container_alias(char *output, int len, const char *prefix, long unsigned int memnodeid)
{
	snprintf(output, len, "%smn%lu", prefix, memnodeid);
	return output;
}

static char *memmanager_container_alias(char *output, int len, const char *prefix, long unsigned int memnodeid)
{
	snprintf(output, len, "%smm%lu", prefix, memnodeid);
	return output;
}

static char *thread_container_alias(char *output, int len, const char *prefix, long unsigned int threadid)
{
	snprintf(output, len, "%st%lu", prefix, threadid);
	return output;
}

static char *worker_container_alias(char *output, int len, const char *prefix, long unsigned int workerid)
{
	snprintf(output, len, "%sw%lu", prefix, workerid);
	return output;
}

static char *mpicommthread_container_alias(char *output, int len, const char *prefix)
{
	snprintf(output, len, "%smpict", prefix);
	return output;
}

static char *program_container_alias(char *output, int len, const char *prefix)
{
	snprintf(output, len, "%sp", prefix);
	return output;
}

static char *scheduler_container_alias(char *output, int len, const char *prefix)
{
	snprintf(output, len, "%ssched", prefix);
	return output;
}
#endif

static void memnode_set_state(double time, const char *prefix, unsigned int memnodeid, const char *name)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	memmanager_container_alias(container, STARPU_POTI_STR_LEN, prefix, memnodeid);
	poti_SetState(time, container, "MS", name);
#else
	fprintf(out_paje_file, "10	%.9f	%smm%u	MS	%s\n", time, prefix, memnodeid, name);
#endif
}

static void worker_set_state(double time, const char *prefix, long unsigned int workerid, const char *name)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, workerid);
	poti_SetState(time, container, "WS", name);
#else
	fprintf(out_paje_file, "10	%.9f	%sw%lu	WS	%s\n", time, prefix, workerid, name);
#endif
}

static void worker_push_state(double time, const char *prefix, long unsigned int workerid, const char *name)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, workerid);
	poti_PushState(time, container, "WS", name);
#else
	fprintf(out_paje_file, "11	%.9f	%sw%lu	WS	%s\n", time, prefix, workerid, name);
#endif
}

static void worker_pop_state(double time, const char *prefix, long unsigned int workerid)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, workerid);
	poti_PopState(time, container, "WS");
#else
	fprintf(out_paje_file, "12	%.9f	%sw%lu	WS\n", time, prefix, workerid);
#endif
}

static void thread_set_state(double time, const char *prefix, long unsigned int threadid, const char *name)
{
	if (find_sync(threadid))
		/* Unless using worker sets, collapse thread and worker */
		return worker_set_state(time, prefix, find_worker_id(threadid), name);

#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
	poti_SetState(time, container, "S", name);
#else
	fprintf(out_paje_file, "10	%.9f	%st%lu	S	%s\n", time, prefix, threadid, name);
#endif
}

static void thread_push_state(double time, const char *prefix, long unsigned int threadid, const char *name)
{
	if (find_sync(threadid))
		/* Unless using worker sets, collapse thread and worker */
		return worker_push_state(time, prefix, find_worker_id(threadid), name);

#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
	poti_PushState(time, container, "S", name);
#else
	fprintf(out_paje_file, "11	%.9f	%st%lu	S	%s\n", time, prefix, threadid, name);
#endif
}

static void thread_pop_state(double time, const char *prefix, long unsigned int threadid)
{
	if (find_sync(threadid))
		/* Unless using worker sets, collapse thread and worker */
		return worker_pop_state(time, prefix, find_worker_id(threadid));

#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
	poti_PopState(time, container, "S");
#else
	fprintf(out_paje_file, "12	%.9f	%st%lu	S\n", time, prefix, threadid);
#endif
}

#ifdef STARPU_ENABLE_PAJE_CODELET_DETAILS
static void worker_set_detailed_state(double time, const char *prefix, long unsigned int workerid, const char *name, unsigned long size, const char *parameters, unsigned long footprint, unsigned long long tag, unsigned long job_id)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, workerid);
	/* TODO: set detailed state */
	poti_SetState(time, container, "WS", name);
#else
	fprintf(out_paje_file, "20	%.9f	%sw%lu	WS	%s	%lu	%s	%08lx	%016llx	%lu\n", time, prefix, workerid, name, size, parameters, footprint, tag, job_id);
#endif
}
#endif

static void mpicommthread_set_state(double time, const char *prefix, const char *name)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	mpicommthread_container_alias(container, STARPU_POTI_STR_LEN, prefix);
	poti_SetState(time, container, "CtS", name);
#else
	fprintf(out_paje_file, "10	%.9f	%smpict	CtS 	%s\n", time, prefix, name);
#endif
}


/*
 *	Initialization
 */

static void handle_new_mem_node(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char program_container[STARPU_POTI_STR_LEN];
		program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
		char new_memnode_container_alias[STARPU_POTI_STR_LEN], new_memnode_container_name[STARPU_POTI_STR_LEN];
		char new_memmanager_container_alias[STARPU_POTI_STR_LEN], new_memmanager_container_name[STARPU_POTI_STR_LEN];
		memnode_container_alias (new_memnode_container_alias, STARPU_POTI_STR_LEN, prefix, ev->param[0]);
		/* TODO: ramkind */
		snprintf(new_memnode_container_name, STARPU_POTI_STR_LEN, "%sMEMNODE%"PRIu64"", prefix, ev->param[0]);
		poti_CreateContainer(get_event_time_stamp(ev, options), new_memnode_container_alias, "Mn", program_container, new_memnode_container_name);

		memmanager_container_alias (new_memmanager_container_alias, STARPU_POTI_STR_LEN, prefix, ev->param[0]);
		/* TODO: ramkind */
		snprintf(new_memmanager_container_name, STARPU_POTI_STR_LEN, "%sMEMMANAGER%"PRIu64"", prefix, ev->param[0]);
		poti_CreateContainer(get_event_time_stamp(ev, options), new_memmanager_container_alias, "Mm", new_memnode_container_alias, new_memmanager_container_name);
#else
		fprintf(out_paje_file, "7	%.9f	%smn%"PRIu64"	Mn	%sp	%sMEMNODE%"PRIu64"\n", get_event_time_stamp(ev, options), prefix, ev->param[0], prefix, options->file_prefix, ev->param[0]);
		fprintf(out_paje_file, "7	%.9f	%smm%"PRIu64"	Mm	%smn%"PRIu64"	%sMEMMANAGER%"PRIu64"\n", get_event_time_stamp(ev, options), prefix, ev->param[0], prefix, ev->param[0], options->file_prefix, ev->param[0]);
#endif

		if (!options->no_bus)
#ifdef STARPU_HAVE_POTI
			poti_SetVariable(get_event_time_stamp(ev, options), new_memmanager_container_alias, "bw", get_event_time_stamp(ev, options));
#else
			fprintf(out_paje_file, "13	%.9f	%smm%"PRIu64"	bw	0.0\n", get_event_time_stamp(ev, options), prefix, ev->param[0]);
#endif
	}
}

static void handle_worker_init_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	/*
	   arg0 : type of worker (cuda, cpu ..)
	   arg1 : memory node
	   arg2 : thread id
	*/
	char *prefix = options->file_prefix;

	int devid = ev->param[2];
	int workerid = ev->param[1];
	int nodeid = ev->param[3];
	int bindid = ev->param[4];
	int set = ev->param[5];
	int threadid = ev->param[6];
	int new_thread;

	new_thread = register_worker_id(threadid, workerid, set);

	char *kindstr = "";
	struct starpu_perfmodel_arch arch;
	arch.ndevices = 1;
	arch.devices = (struct starpu_perfmodel_device *)malloc(sizeof(struct starpu_perfmodel_device));

	switch (ev->param[0])
	{
		case _STARPU_FUT_APPS_KEY:
			set_next_other_worker_color(workerid);
			kindstr = "APPS";
			break;
		case _STARPU_FUT_CPU_KEY:
			set_next_cpu_worker_color(workerid);
			kindstr = "CPU";
			arch.devices[0].type = STARPU_CPU_WORKER;
			arch.devices[0].devid = 0;
			arch.devices[0].ncores = 1;
			break;
		case _STARPU_FUT_CUDA_KEY:
			set_next_cuda_worker_color(workerid);
			kindstr = "CUDA";
			arch.devices[0].type = STARPU_CUDA_WORKER;
			arch.devices[0].devid = devid;
			arch.devices[0].ncores = 1;
			break;
		case _STARPU_FUT_OPENCL_KEY:
			set_next_opencl_worker_color(workerid);
			kindstr = "OPENCL";
			arch.devices[0].type = STARPU_OPENCL_WORKER;
			arch.devices[0].devid = devid;
			arch.devices[0].ncores = 1;
			break;
		case _STARPU_FUT_MIC_KEY:
			set_next_mic_worker_color(workerid);
			kindstr = "mic";
			arch.devices[0].type = STARPU_MIC_WORKER;
			arch.devices[0].devid = devid;
			arch.devices[0].ncores = 1;
			break;
		case _STARPU_FUT_SCC_KEY:
			set_next_scc_worker_color(workerid);
			kindstr = "scc";
			arch.devices[0].type = STARPU_SCC_WORKER;
			arch.devices[0].devid = devid;
			arch.devices[0].ncores = 1;
			break;
		default:
			STARPU_ABORT();
	}

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char new_thread_container_alias[STARPU_POTI_STR_LEN];
		thread_container_alias (new_thread_container_alias, STARPU_POTI_STR_LEN, prefix, threadid);
		char new_worker_container_alias[STARPU_POTI_STR_LEN];
		worker_container_alias (new_worker_container_alias, STARPU_POTI_STR_LEN, prefix, workerid);
		char memnode_container[STARPU_POTI_STR_LEN];
		memnode_container_alias(memnode_container, STARPU_POTI_STR_LEN, prefix, nodeid);
		char new_thread_container_name[STARPU_POTI_STR_LEN];
		snprintf(new_thread_container_name, STARPU_POTI_STR_LEN, "%s%d", prefix, bindid);
		char new_worker_container_name[STARPU_POTI_STR_LEN];
		snprintf(new_worker_container_name, STARPU_POTI_STR_LEN, "%s%s%d", prefix, kindstr, devid);
		if (new_thread)
			poti_CreateContainer(get_event_time_stamp(ev, options), new_thread_container_alias, "T", memnode_container, new_thread_container_name);
		poti_CreateContainer(get_event_time_stamp(ev, options), new_worker_container_alias, "W", new_thread_container_alias, new_worker_container_name);
#else
		if (new_thread)
			fprintf(out_paje_file, "7	%.9f	%st%d	T	%smn%d	%s%d\n",
				get_event_time_stamp(ev, options), prefix, threadid, prefix, nodeid, prefix, bindid);
		fprintf(out_paje_file, "7	%.9f	%sw%d	W	%st%d	%s%s%d\n",
			get_event_time_stamp(ev, options), prefix, workerid, prefix, threadid, prefix, kindstr, devid);
#endif
	}

	/* start initialization */
	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), prefix, threadid, "In");

	if (activity_file)
		fprintf(activity_file, "name\t%d\t%s %d\n", workerid, kindstr, devid);

	snprintf(options->worker_names[workerid], 256, "%s %d", kindstr, devid);
	options->worker_archtypes[workerid] = arch;
}

static void handle_worker_init_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	int worker;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), prefix, ev->param[0], "B");

	if (ev->nb_params < 2)
		worker = find_worker_id(ev->param[0]);
	else
		worker = ev->param[1];

	if (out_paje_file)
		worker_set_state(get_event_time_stamp(ev, options), prefix, worker, "I");

	/* Initilize the accumulated time counters */
	last_activity_flush_timestamp[worker] = get_event_time_stamp(ev, options);
	accumulated_sleep_time[worker] = 0.0;
	accumulated_exec_time[worker] = 0.0;
}

static void handle_worker_deinit_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), prefix, ev->param[0], "D");
}

static void handle_worker_deinit_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char worker_container[STARPU_POTI_STR_LEN];
		thread_container_alias(worker_container, STARPU_POTI_STR_LEN, prefix, ev->param[1]);
		poti_DestroyContainer(get_event_time_stamp(ev, options), "T", worker_container);
#else
		fprintf(out_paje_file, "8	%.9f	%st%"PRIu64"	T\n",
			get_event_time_stamp(ev, options), prefix, ev->param[1]);
#endif
	}
}

#ifdef STARPU_HAVE_POTI
static void create_paje_state_color(char *name, char *type, float red, float green, float blue)
{
	char color[STARPU_POTI_STR_LEN];
	snprintf(color, STARPU_POTI_STR_LEN, "%f %f %f", red, green, blue);
	poti_DefineEntityValue(name, type, name, color);
}
#endif

static void create_paje_state_if_not_found(char *name, struct starpu_fxt_options *options)
{
	struct _starpu_symbol_name *itor;
	for (itor = _starpu_symbol_name_list_begin(&symbol_list);
		itor != _starpu_symbol_name_list_end(&symbol_list);
		itor = _starpu_symbol_name_list_next(itor))
	{
		if (!strcmp(name, itor->name))
		{
			/* we found an entry */
			return;
		}
	}

	/* it's the first time ... */
	struct _starpu_symbol_name *entry = _starpu_symbol_name_new();
	entry->name = malloc(strlen(name) + 1);
	strcpy(entry->name, name);

	_starpu_symbol_name_list_push_front(&symbol_list, entry);

	/* choose some colour ... that's disguting yes */
	unsigned hash_symbol_red = get_colour_symbol_red(name);
	unsigned hash_symbol_green = get_colour_symbol_green(name);
	unsigned hash_symbol_blue = get_colour_symbol_blue(name);

	uint32_t hash_sum = hash_symbol_red + hash_symbol_green + hash_symbol_blue;

	float red, green, blue;
	if (options->per_task_colour)
	{
		red = (1.0f * hash_symbol_red) / hash_sum;
		green = (1.0f * hash_symbol_green) / hash_sum;
		blue = (1.0f * hash_symbol_blue) / hash_sum;
	}
	else
	{
		/* Use the hardcoded value for execution mode */
		red = 0.0f;
		green = 0.6f;
		blue = 0.4f;
	}

	/* create the Paje state */
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		create_paje_state_color(name, "WS", red, green, blue);
		int i;
		for(i = 1; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			char ctx[10];
			snprintf(ctx, sizeof(ctx), "Ctx%d", i);
			if(i%10 == 1)
				create_paje_state_color(name, ctx, 255.0, 102.0, 255.0);
			if(i%10 == 2)
				create_paje_state_color(name, ctx, .0, 255.0, 0.0);
			if(i%10 == 3)
				create_paje_state_color(name, ctx, 255.0, 255.0, .0);
			if(i%10 == 4)
				create_paje_state_color(name, ctx, .0, 245.0, 255.0);
			if(i%10 == 5)
				create_paje_state_color(name, ctx, .0, .0, .0);
			if(i%10 == 6)
				create_paje_state_color(name, ctx, .0, .0, 128.0);
			if(i%10 == 7)
				create_paje_state_color(name, ctx, 105.0, 105.0, 105.0);
			if(i%10 == 8)
				create_paje_state_color(name, ctx, 255.0, .0, 255.0);
			if(i%10 == 9)
				create_paje_state_color(name, ctx, .0, .0, 1.0);
			if(i%10 == 0)
				create_paje_state_color(name, ctx, 154.0, 205.0, 50.0);

		}
/* 		create_paje_state_color(name, "Ctx1", 255.0, 102.0, 255.0); */
/* 		create_paje_state_color(name, "Ctx2", .0, 255.0, 0.0); */
/* 		create_paje_state_color(name, "Ctx3", 255.0, 255.0, .0); */
/* 		create_paje_state_color(name, "Ctx4", .0, 245.0, 255.0); */
/* 		create_paje_state_color(name, "Ctx5", .0, .0, .0); */
/* 		create_paje_state_color(name, "Ctx6", .0, .0, 128.0); */
/* 		create_paje_state_color(name, "Ctx7", 105.0, 105.0, 105.0); */
/* 		create_paje_state_color(name, "Ctx8", 255.0, .0, 255.0); */
/* 		create_paje_state_color(name, "Ctx9", .0, .0, 1.0); */
/* 		create_paje_state_color(name, "Ctx10", 154.0, 205.0, 50.0); */
#else
		fprintf(out_paje_file, "6	%s	WS	%s	\"%f %f %f\" \n", name, name, red, green, blue);
		int i;
		for(i = 1; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			if(i%10 == 1)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\"255.0 102.0 255.0\" \n", name, i, name);
			if(i%10 == 2)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\".0 255.0 .0\" \n", name, i, name);
			if(i%10 == 3)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\"225.0 225.0 .0\" \n", name, i, name);
			if(i%10 == 4)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\".0 245.0 255.0\" \n", name, i, name);
			if(i%10 == 5)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\".0 .0 .0\" \n", name, i, name);
			if(i%10 == 6)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\".0 .0 128.0\" \n", name, i, name);
			if(i%10 == 7)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\"105.0 105.0 105.0\" \n", name, i, name);
			if(i%10 == 8)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\"255.0 .0 255.0\" \n", name, i, name);
			if(i%10 == 9)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\".0 .0 1.0\" \n", name, i, name);
			if(i%10 == 0)
				fprintf(out_paje_file, "6	%s	Ctx%d	%s	\"154.0 205.0 50.0\" \n", name, i, name);

		}

/* 		fprintf(out_paje_file, "6	%s	Ctx1	%s	\"255.0 102.0 255.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx2	%s	\".0 255.0 .0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx3	%s	\"225.0 225.0 .0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx4	%s	\".0 245.0 255.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx5	%s	\".0 .0 .0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx6	%s	\".0 .0 128.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx7	%s	\"105.0 105.0 105.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx8	%s	\"255.0 .0 255.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx9	%s	\".0 .0 1.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx10	%s	\"154.0 205.0 50.0\" \n", name, name); */
#endif
	}

}


static void handle_start_codelet_body(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker = ev->param[2];

	if (worker < 0) return;

	unsigned long has_name = ev->param[3];
	char *name = has_name?(char *)&ev->param[4]:"unknown";

	snprintf(_starpu_last_codelet_symbol[worker], sizeof(_starpu_last_codelet_symbol[worker]), "%s", name);
	last_codelet_parameter[worker] = 0;

	double start_codelet_time = get_event_time_stamp(ev, options);
	last_codelet_start[worker] = start_codelet_time;

	create_paje_state_if_not_found(name, options);

	struct task_info *task = get_task(ev->param[0]);
	task->start_time = start_codelet_time;
	task->workerid = worker;
	task->name = strdup(name);

#ifndef STARPU_ENABLE_PAJE_CODELET_DETAILS
	if (out_paje_file)
	{
		char *prefix = options->file_prefix;
		unsigned sched_ctx = ev->param[1];

		worker_set_state(start_codelet_time, prefix, ev->param[2], name);
		if (sched_ctx != 0)
		{
#ifdef STARPU_HAVE_POTI
			char container[STARPU_POTI_STR_LEN];
			char ctx[6];
			snprintf(ctx, sizeof(ctx), "Ctx%d", sched_ctx);
			worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, ev->param[2]);
			poti_SetState(start_codelet_time, container, ctx, name);
#else
			fprintf(out_paje_file, "10	%.9f	%sw%"PRIu64"	Ctx%d	%s\n", start_codelet_time, prefix, ev->param[2], sched_ctx, name);
#endif
		}
	}
#endif /* STARPU_ENABLE_PAJE_CODELET_DETAILS */

}

static void handle_model_name(struct fxt_ev_64 *ev, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	struct task_info *task = get_task(ev->param[0]);
	char *name = (char *)&ev->param[1];
	task->model_name = strdup(name);
}

static void handle_codelet_data(struct fxt_ev_64 *ev STARPU_ATTRIBUTE_UNUSED, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	int worker = ev->param[0];
	if (worker < 0) return;
	int num = last_codelet_parameter[worker]++;
	if (num >= MAX_PARAMETERS)
		return;
	snprintf(last_codelet_parameter_description[worker][num], sizeof(last_codelet_parameter_description[worker][num]), "%s", (char*) &ev->param[1]);
}

static void handle_codelet_data_handle(struct fxt_ev_64 *ev STARPU_ATTRIBUTE_UNUSED, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	struct task_info *task = get_task(ev->param[0]);
	unsigned alloc = 0;

	if (task->ndata == 0)
		/* Start with 8=2^3, should be plenty in most cases */
		alloc = 8;
	else if (task->ndata >= 8)
	{
		/* Allocate dependencies array by powers of two */
		if (! ((task->ndata - 1) & task->ndata)) /* Is task->ndata a power of two? */
		{
			/* We have filled the previous power of two, get another one */
			alloc = task->ndata * 2;
		}
	}
	if (alloc)
		task->data = realloc(task->data, sizeof(*task->data) * alloc);
	task->data[task->ndata].handle = ev->param[1];
	task->data[task->ndata].size = ev->param[2];
	task->data[task->ndata].mode = ev->param[3];
	task->ndata++;
}

static void handle_codelet_details(struct fxt_ev_64 *ev STARPU_ATTRIBUTE_UNUSED, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	int worker = ev->param[5];
	unsigned long job_id = ev->param[6];

	if (worker < 0) return;

	int i;
	char parameters[256];
	size_t eaten = 0;
	if (!last_codelet_parameter[worker])
		eaten += snprintf(parameters + eaten, sizeof(parameters) - eaten, "nodata");
	else
	for (i = 0; i < last_codelet_parameter[worker] && i < MAX_PARAMETERS; i++)
	{
		eaten += snprintf(parameters + eaten, sizeof(parameters) - eaten, "%s%s", i?"_":"", last_codelet_parameter_description[worker][i]);
	}

	struct task_info *task = get_task(job_id);
	task->parameters = strdup(parameters);
	task->footprint = ev->param[3];
	task->tag = ev->param[4];

	if (out_paje_file)
	{

#ifdef STARPU_ENABLE_PAJE_CODELET_DETAILS
		char *prefix = options->file_prefix;
		unsigned sched_ctx = ev->param[1];

		worker_set_detailed_state(last_codelet_start[worker], prefix, worker, _starpu_last_codelet_symbol[worker], ev->param[2], parameters, ev->param[3], ev->param[4], job_id);
		if (sched_ctx != 0)
		{
#ifdef STARPU_HAVE_POTI
			char container[STARPU_POTI_STR_LEN];
			char ctx[6];
			snprintf(ctx, sizeof(ctx), "Ctx%d", sched_ctx);
			worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, ev->param[5]);
			poti_SetState(last_codelet_start[worker], container, ctx, _starpu_last_codelet_symbol[worker]);
#else
			fprintf(out_paje_file, "20	%.9f	%sw%"PRIu64"	Ctx%d	%s	%lu	%s	%08lx	%016llx	%lu\n", last_codelet_start[worker], prefix, ev->param[2], sched_ctx, _starpu_last_codelet_symbol[worker], (unsigned long) ev->param[2], parameters, (unsigned long) ev->param[3], (unsigned long long) ev->param[4], job_id);
#endif
		}
#endif /* STARPU_ENABLE_PAJE_CODELET_DETAILS */
	}
}

static long dumped_codelets_count;
static struct starpu_fxt_codelet_event *dumped_codelets;

static void handle_end_codelet_body(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker = ev->param[3];
	if (worker < 0) return;

	char *prefix = options->file_prefix;

	double end_codelet_time = get_event_time_stamp(ev, options);

	size_t codelet_size = ev->param[1];
	uint32_t codelet_hash = ev->param[2];

	if (out_paje_file)
		worker_set_state(end_codelet_time, prefix, worker, "I");

	double codelet_length = (end_codelet_time - last_codelet_start[worker]);

	get_task(ev->param[0])->end_time = end_codelet_time;

	update_accumulated_time(worker, 0.0, codelet_length, end_codelet_time, 0);

	if (distrib_time)
	     fprintf(distrib_time, "%s\t%s%d\t%ld\t%"PRIx32"\t%.9f\n", _starpu_last_codelet_symbol[worker],
		     prefix, worker, (unsigned long) codelet_size, codelet_hash, codelet_length);

	if (options->dumped_codelets)
	{
		dumped_codelets_count++;
		dumped_codelets = realloc(dumped_codelets, dumped_codelets_count*sizeof(struct starpu_fxt_codelet_event));

		snprintf(dumped_codelets[dumped_codelets_count - 1].symbol, 256, "%s", _starpu_last_codelet_symbol[worker]);
		dumped_codelets[dumped_codelets_count - 1].workerid = worker;
		snprintf(dumped_codelets[dumped_codelets_count - 1].perfmodel_archname, 256, "%s", (char *)&ev->param[4]);
		dumped_codelets[dumped_codelets_count - 1].size = codelet_size;
		dumped_codelets[dumped_codelets_count - 1].hash = codelet_hash;
		dumped_codelets[dumped_codelets_count - 1].time = codelet_length;
	}
	_starpu_last_codelet_symbol[worker][0] = 0;
}

static void handle_start_executing(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	int threadid = ev->param[0];

	if (out_paje_file && !find_sync(threadid))
		thread_set_state(get_event_time_stamp(ev, options), prefix, threadid, "E");
}

static void handle_end_executing(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	int threadid = ev->param[0];

	if (out_paje_file && !find_sync(threadid))
		thread_set_state(get_event_time_stamp(ev, options), prefix, threadid, "B");
}

static void handle_user_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	unsigned long code = ev->param[0];
#ifdef STARPU_HAVE_POTI
	char paje_value[STARPU_POTI_STR_LEN], container[STARPU_POTI_STR_LEN];
	snprintf(paje_value, STARPU_POTI_STR_LEN, "%lu", code);
#endif

	char *prefix = options->file_prefix;

	worker = find_worker_id(ev->param[1]);
	if (worker < 0)
	{
		if (out_paje_file)
#ifdef STARPU_HAVE_POTI
			program_container_alias (container, STARPU_POTI_STR_LEN, prefix);
#else
			fprintf(out_paje_file, "9	%.9f	user_event	%sp	%lu\n", get_event_time_stamp(ev, options), prefix, code);
#endif
	}
	else
	{
		if (out_paje_file)
#ifdef STARPU_HAVE_POTI
			thread_container_alias (container, STARPU_POTI_STR_LEN, prefix, ev->param[1]);
#else
			fprintf(out_paje_file, "9	%.9f	user_event	%st%"PRIu64"	%lu\n", get_event_time_stamp(ev, options), prefix, ev->param[1], code);
#endif
	}
#ifdef STARPU_HAVE_POTI
	if (out_paje_file)
		poti_NewEvent(get_event_time_stamp(ev, options), container, "user_event", paje_value);
#endif
}

static void handle_start_callback(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[1]);
	if (worker < 0)
		return;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], "C");
}

static void handle_end_callback(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[1]);
	if (worker < 0)
		return;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], "B");
}

static void handle_hypervisor_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0)
		return;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "H");
}

static void handle_hypervisor_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0)
		return;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "B");
}

static void handle_worker_status(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *newstatus)
{
	int worker;
	worker = find_worker_id(ev->param[1]);
	if (worker < 0)
		return;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], newstatus);
}

static double last_sleep_start[STARPU_NMAXWORKERS];

static void handle_worker_scheduling_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0) return;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "Sc");
}

static void handle_worker_scheduling_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0) return;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "B");
}

static void handle_worker_scheduling_push(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0) return;

	if (out_paje_file)
		thread_push_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "Sc");
}

static void handle_worker_scheduling_pop(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0) return;

	if (out_paje_file)
		thread_pop_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0]);
}

static void handle_worker_sleep_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0) return;

	double start_sleep_time = get_event_time_stamp(ev, options);
	last_sleep_start[worker] = start_sleep_time;

	if (out_paje_file)
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "Sl");
}

static void handle_worker_sleep_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	worker = find_worker_id(ev->param[0]);
	if (worker < 0) return;

	double end_sleep_timestamp = get_event_time_stamp(ev, options);

	if (out_paje_file)
		thread_set_state(end_sleep_timestamp, options->file_prefix, ev->param[0], "B");

	double sleep_length = end_sleep_timestamp - last_sleep_start[worker];

	update_accumulated_time(worker, sleep_length, 0.0, end_sleep_timestamp, 0);
}

static void handle_data_copy(void)
{
}

static const char *copy_link_type(unsigned prefetch)
{
	switch (prefetch)
	{
		case 0: return "F";
		case 1: return "PF";
		case 2: return "IF";
		default: STARPU_ASSERT(0);
	}
}

static void handle_start_driver_copy(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned src = ev->param[0];
	unsigned dst = ev->param[1];
	unsigned size = ev->param[2];
	unsigned comid = ev->param[3];
	unsigned prefetch = ev->param[4];
	const char *link_type = copy_link_type(prefetch);

	char *prefix = options->file_prefix;

	if (!options->no_bus)
	{
		if (out_paje_file)
		{
			double time = get_event_time_stamp(ev, options);
			memnode_set_state(time, prefix, dst, "Co");
#ifdef STARPU_HAVE_POTI
			char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN], src_memnode_container[STARPU_POTI_STR_LEN];
			char program_container[STARPU_POTI_STR_LEN];
			snprintf(paje_value, STARPU_POTI_STR_LEN, "%u", size);
			snprintf(paje_key, STARPU_POTI_STR_LEN, "com_%u", comid);
			program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
			memmanager_container_alias(src_memnode_container, STARPU_POTI_STR_LEN, prefix, src);
			poti_StartLink(time, program_container, link_type, src_memnode_container, paje_value, paje_key);
#else
			fprintf(out_paje_file, "18	%.9f	%s	%sp	%u	%smm%u	com_%u\n", time, link_type, prefix, size, prefix, src, comid);
#endif
		}

		/* create a structure to store the start of the communication, this will be matched later */
		struct _starpu_communication *com = _starpu_communication_new();
		com->comid = comid;
		com->comm_start = get_event_time_stamp(ev, options);

		com->src_node = src;
		com->dst_node = dst;

		_starpu_communication_list_push_back(&communication_list, com);
	}

}


static void handle_work_stealing(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned dst = ev->param[0];
	unsigned src = ev->param[1];
	unsigned size = 0;

	char *prefix = options->file_prefix;


	if (out_paje_file)
	{
		double time = get_event_time_stamp(ev, options);
#ifdef STARPU_HAVE_POTI
		char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN], src_worker_container[STARPU_POTI_STR_LEN], dst_worker_container[STARPU_POTI_STR_LEN];
		char program_container[STARPU_POTI_STR_LEN];
		snprintf(paje_value, STARPU_POTI_STR_LEN, "%u", size);
		snprintf(paje_key, STARPU_POTI_STR_LEN, "steal_%u", steal_number);
		program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
		worker_container_alias(src_worker_container, STARPU_POTI_STR_LEN, prefix, src);
		worker_container_alias(dst_worker_container, STARPU_POTI_STR_LEN, prefix, dst);
		poti_StartLink(time, program_container, "WSL", src_worker_container, paje_value, paje_key);
		poti_EndLink(time+0.000000001, program_container, "WSL", dst_worker_container, paje_value, paje_key);
#else

		fprintf(out_paje_file, "18	%.9f	WSL	%sp	%u	%sw%d	steal_%u\n", time, prefix, size, prefix, src, steal_number);
		fprintf(out_paje_file, "19	%.9f	WSL	%sp	%u	%sw%d	steal_%u\n", time+0.000000001, prefix, size, prefix, dst, steal_number);
#endif
	}

	steal_number++;
}


static void handle_end_driver_copy(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned dst = ev->param[1];
	unsigned size = ev->param[2];
	unsigned comid = ev->param[3];
	unsigned prefetch = ev->param[4];
	const char *link_type = copy_link_type(prefetch);

	char *prefix = options->file_prefix;

	if (!options->no_bus)
	{
		if (out_paje_file)
		{
			double time = get_event_time_stamp(ev, options);
			memnode_set_state(time, prefix, dst, "No");
#ifdef STARPU_HAVE_POTI
			char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN];
			char dst_memnode_container[STARPU_POTI_STR_LEN], program_container[STARPU_POTI_STR_LEN];
			snprintf(paje_value, STARPU_POTI_STR_LEN, "%u", size);
			snprintf(paje_key, STARPU_POTI_STR_LEN, "com_%u", comid);
			program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
			memmanager_container_alias(dst_memnode_container, STARPU_POTI_STR_LEN, prefix, dst);
			poti_EndLink(time, program_container, link_type, dst_memnode_container, paje_value, paje_key);
#else
			fprintf(out_paje_file, "19	%.9f	%s	%sp	%u	%smm%u	com_%u\n", time, link_type, prefix, size, prefix, dst, comid);
#endif
		}

		/* look for a data transfer to match */
		struct _starpu_communication *itor;
		for (itor = _starpu_communication_list_begin(&communication_list);
			itor != _starpu_communication_list_end(&communication_list);
			itor = _starpu_communication_list_next(itor))
		{
			if (itor->comid == comid)
			{
				double comm_end = get_event_time_stamp(ev, options);
				double bandwidth = (double)((0.001*size)/(comm_end - itor->comm_start));

				itor->bandwidth = bandwidth;

				struct _starpu_communication *com = _starpu_communication_new();
				com->comid = comid;
				com->comm_start = get_event_time_stamp(ev, options);
				com->bandwidth = -bandwidth;

				com->src_node = itor->src_node;
				com->dst_node = itor->dst_node;

				_starpu_communication_list_push_back(&communication_list, com);

				break;
			}
		}
	}
}

static void handle_start_driver_copy_async(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned dst = ev->param[1];

	char *prefix = options->file_prefix;

	if (!options->no_bus)
		if (out_paje_file)
			memnode_set_state(get_event_time_stamp(ev, options), prefix, dst, "CoA");

}

static void handle_end_driver_copy_async(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned dst = ev->param[1];

	char *prefix = options->file_prefix;

	if (!options->no_bus)
		if (out_paje_file)
			memnode_set_state(get_event_time_stamp(ev, options), prefix, dst, "Co");
}

static void handle_memnode_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];

	if (out_paje_file)
		memnode_set_state(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr);
}

/*
 *	Number of task submitted to the scheduler
 */
static int curq_size = 0;
static int nsubmitted = 0;

static void handle_job_push(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double current_timestamp = get_event_time_stamp(ev, options);

	curq_size++;

	_starpu_fxt_component_update_ntasks(nsubmitted, curq_size);

	if (!options->no_counter && out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];

		scheduler_container_alias(container, STARPU_POTI_STR_LEN, options->file_prefix);
		poti_SetVariable(current_timestamp, container, "nready", (double)curq_size);
#else
		fprintf(out_paje_file, "13	%.9f	%ssched	nready	%f\n", current_timestamp, options->file_prefix, (float)curq_size);
#endif
	}

	if (activity_file)
	fprintf(activity_file, "cnt_ready\t%.9f\t%d\n", current_timestamp, curq_size);
}

static void handle_job_pop(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double current_timestamp = get_event_time_stamp(ev, options);

	curq_size--;
	nsubmitted--;
	_starpu_fxt_component_update_ntasks(nsubmitted, curq_size);

	if (!options->no_counter && out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];
		scheduler_container_alias(container, STARPU_POTI_STR_LEN, options->file_prefix);
		poti_SetVariable(current_timestamp, container, "nready", (double)curq_size);
	poti_SetVariable(current_timestamp, container, "nsubmitted", (double)nsubmitted);
#else
		fprintf(out_paje_file, "13	%.9f	%ssched	nready	%f\n", current_timestamp, options->file_prefix, (float)curq_size);
		fprintf(out_paje_file, "13	%.9f	%ssched	nsubmitted	%f\n", current_timestamp, options->file_prefix, (float)nsubmitted);
#endif
	}

	if (activity_file)
	{
		fprintf(activity_file, "cnt_ready\t%.9f\t%d\n", current_timestamp, curq_size);
		fprintf(activity_file, "cnt_submitted\t%.9f\t%d\n", current_timestamp, nsubmitted);
	}
}

static void handle_component_new(struct fxt_ev_64 *ev, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	_starpu_fxt_component_new(ev->param[0], (char *)&ev->param[1]);
}

static void handle_component_connect(struct fxt_ev_64 *ev, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	_starpu_fxt_component_connect(ev->param[0], ev->param[1]);
}

static void handle_component_push(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double current_timestamp = get_event_time_stamp(ev, options);
	int workerid = find_worker_id(ev->param[0]);
	_starpu_fxt_component_push(anim_file, options, current_timestamp, workerid, ev->param[1], ev->param[2], ev->param[3], ev->param[4]);
}

static void handle_component_pull(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double current_timestamp = get_event_time_stamp(ev, options);
	int workerid = find_worker_id(ev->param[0]);
	_starpu_fxt_component_pull(anim_file, options, current_timestamp, workerid, ev->param[1], ev->param[2], ev->param[3], ev->param[4]);
}

static
void handle_update_task_cnt(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double current_timestamp = get_event_time_stamp(ev, options);

	nsubmitted++;
	_starpu_fxt_component_update_ntasks(nsubmitted, curq_size);
	if (!options->no_counter && out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];
		scheduler_container_alias(container, STARPU_POTI_STR_LEN, options->file_prefix);
		poti_SetVariable(current_timestamp, container, "nsubmitted", (double)nsubmitted);
#else
		fprintf(out_paje_file, "13	%.9f	%ssched nsubmitted	%f\n", current_timestamp, options->file_prefix, (float)nsubmitted);
#endif
	}


	if (activity_file)
		fprintf(activity_file, "cnt_submitted\t%.9f\t%d\n", current_timestamp, nsubmitted);
}

static void handle_tag(struct fxt_ev_64 *ev)
{
	uint64_t tag;
	unsigned long job;

	tag = ev->param[0];
	job = ev->param[1];

	_starpu_fxt_dag_add_tag(tag, job);
}

static void handle_tag_deps(struct fxt_ev_64 *ev)
{
	uint64_t child;
	uint64_t father;

	child = ev->param[0];
	father = ev->param[1];

	_starpu_fxt_dag_add_tag_deps(child, father);
}

static void handle_task_deps(struct fxt_ev_64 *ev)
{
	unsigned long dep_prev = ev->param[0];
	unsigned long dep_succ = ev->param[1];

	struct task_info *task = get_task(dep_succ);
	unsigned alloc = 0;

	if (task->ndeps == 0)
		/* Start with 8=2^3, should be plenty in most cases */
		alloc = 8;
	else if (task->ndeps >= 8)
	{
		/* Allocate dependencies array by powers of two */
		if (! ((task->ndeps - 1) & task->ndeps)) /* Is task->ndeps a power of two? */
		{
			/* We have filled the previous power of two, get another one */
			alloc = task->ndeps * 2;
		}
	}
	if (alloc)
		task->dependencies = realloc(task->dependencies, sizeof(*task->dependencies) * alloc);
	task->dependencies[task->ndeps++] = dep_prev;

	/* There is a dependency between both job id : dep_prev -> dep_succ */
	_starpu_fxt_dag_add_task_deps(dep_prev, dep_succ);
}

static void handle_task_submit(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long job_id;
	job_id = ev->param[0];

	get_task(job_id)->submit_time = get_event_time_stamp(ev, options);
}

static void handle_task_done(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long job_id;
	job_id = ev->param[0];

	unsigned long has_name = ev->param[3];
	char *name = has_name?(char *)&ev->param[4]:"unknown";

        int worker;
        worker = find_worker_id(ev->param[1]);

	const char *colour;
	char buffer[32];
	if (options->per_task_colour)
	{
		snprintf(buffer, 32, "#%x%x%x",
			 get_colour_symbol_red(name)/4,
			 get_colour_symbol_green(name)/4,
			 get_colour_symbol_blue(name)/4);
		colour = &buffer[0];
	}
	else
	{
		colour= (worker < 0)?"#aaaaaa":get_worker_color(worker);
	}

	unsigned exclude_from_dag = ev->param[2];
	get_task(job_id)->exclude_from_dag = exclude_from_dag;
	task_dump(job_id);

	if (!exclude_from_dag)
		_starpu_fxt_dag_set_task_done(job_id, name, colour);
}

static void handle_tag_done(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	uint64_t tag_id;
	tag_id = ev->param[0];

	unsigned long has_name = ev->param[2];
	char *name = has_name?(char *)&ev->param[3]:"unknown";

        int worker;
        worker = find_worker_id(ev->param[1]);

	const char *colour;
	char buffer[32];
	if (options->per_task_colour)
	{
		snprintf(buffer, 32, "%.4f,%.4f,%.4f",
			 get_colour_symbol_red(name)/1024.0,
			 get_colour_symbol_green(name)/1024.0,
			 get_colour_symbol_blue(name)/1024.0);
		colour = &buffer[0];
	}
	else
	{
		colour= (worker < 0)?"white":get_worker_color(worker);
	}

	_starpu_fxt_dag_set_tag_done(tag_id, colour);
}

static void handle_mpi_barrier(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int rank = ev->param[0];

	STARPU_ASSERT(rank == options->file_rank || options->file_rank == -1);

	/* Add an event in the trace */
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN], paje_value[STARPU_POTI_STR_LEN];
		snprintf(container, STARPU_POTI_STR_LEN, "%sp", options->file_prefix);
		snprintf(paje_value, STARPU_POTI_STR_LEN, "%d", rank);
		poti_NewEvent(get_event_time_stamp(ev, options), container, "prog_event", paje_value);
#else
		fprintf(out_paje_file, "9	%.9f	prog_event	%sp	%d\n", get_event_time_stamp(ev, options), options->file_prefix, rank);
#endif
	}
}

static void handle_mpi_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	char *prefix = options->file_prefix;

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char program_container[STARPU_POTI_STR_LEN];
		program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
		char new_mpicommthread_container_alias[STARPU_POTI_STR_LEN], new_mpicommthread_container_name[STARPU_POTI_STR_LEN];
		mpicommthread_container_alias(new_mpicommthread_container_alias, STARPU_POTI_STR_LEN, prefix);
		snprintf(new_mpicommthread_container_alias, STARPU_POTI_STR_LEN, "%smpict", prefix);
		poti_CreateContainer(date, new_mpicommthread_container_alias, "MPICt", program_container, new_mpicommthread_container_name);
#else
		fprintf(out_paje_file, "7	%.9f	%smpict		MPICt	%sp	%smpict\n", date, prefix, prefix, prefix);
#endif
		mpicommthread_set_state(date, prefix, "Sl");
	}
}

static void handle_mpi_stop(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	char *prefix = options->file_prefix;

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char mpicommthread_container[STARPU_POTI_STR_LEN];
		mpicommthread_container_alias(mpicommthread_container, STARPU_POTI_STR_LEN, prefix);
		poti_DestroyContainer(date, "MPICt", mpicommthread_container);
#else
		fprintf(out_paje_file, "8	%.9f	%smpict		MPICt\n",
			date, prefix);
#endif
	}
}

static void handle_mpi_isend_submit_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "SdS");
}

static void handle_mpi_isend_submit_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int dest = ev->param[0];
	int mpi_tag = ev->param[1];
	size_t size = ev->param[2];
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");

	_starpu_fxt_mpi_add_send_transfer(options->file_rank, dest, mpi_tag, size, date);
}

static void handle_mpi_irecv_submit_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "RvS");
}

static void handle_mpi_irecv_submit_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_mpi_isend_complete_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "SdC");
}

static void handle_mpi_isend_complete_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_mpi_irecv_complete_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int src = ev->param[0];
	int mpi_tag = ev->param[1];
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "RvC");

	_starpu_fxt_mpi_add_recv_transfer(src, options->file_rank, mpi_tag, date);
}

static void handle_mpi_irecv_complete_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_mpi_sleep_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "Sl");
}

static void handle_mpi_sleep_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_mpi_dtesting_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "DT");
}

static void handle_mpi_dtesting_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_mpi_utesting_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "UT");
}

static void handle_mpi_utesting_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_mpi_uwait_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "UW");
}

static void handle_mpi_uwait_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_set_profiling(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int status = ev->param[0];

	if (activity_file)
	fprintf(activity_file, "set_profiling\t%.9f\t%d\n", get_event_time_stamp(ev, options), status);
}

static void handle_task_wait_for_all(void)
{
	_starpu_fxt_dag_add_sync_point();
}

static void handle_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *event = (char*)&ev->param[0];
	/* Add an event in the trace */
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];
		snprintf(container, STARPU_POTI_STR_LEN, "%sp", options->file_prefix);
		poti_NewEvent(get_event_time_stamp(ev, options), container, "prog_event", event);
#else
		fprintf(out_paje_file, "9	%.9f	prog_event	%sp	%s\n", get_event_time_stamp(ev, options), options->file_prefix, event);
#endif
	}
}

static void handle_thread_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	/* Add an event in the trace */
	if (out_paje_file)
	{
		char *event = (char*)&ev->param[1];

#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];
		thread_container_alias(container, STARPU_POTI_STR_LEN, options->file_prefix, ev->param[0]);
		poti_NewEvent(get_event_time_stamp(ev, options), container, "thread_event", event);
#else
		fprintf(out_paje_file, "9	%.9f	thread_event	%st%"PRIu64"	%s\n", get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], event);
#endif
	}
}

static
void _starpu_fxt_display_bandwidth(struct starpu_fxt_options *options)
{
	float current_bandwidth_per_node[STARPU_MAXNODES] = {0.0};

	char *prefix = options->file_prefix;

	struct _starpu_communication*itor;
	for (itor = _starpu_communication_list_begin(&communication_list);
		itor != _starpu_communication_list_end(&communication_list);
		itor = _starpu_communication_list_next(itor))
	{
		current_bandwidth_per_node[itor->src_node] +=  itor->bandwidth;
		if (out_paje_file)
		{
#ifdef STARPU_HAVE_POTI
			char src_memnode_container[STARPU_POTI_STR_LEN];
			memmanager_container_alias(src_memnode_container, STARPU_POTI_STR_LEN, prefix, itor->src_node);
			poti_SetVariable(itor->comm_start, src_memnode_container, "bw", current_bandwidth_per_node[itor->src_node]);
#else
			fprintf(out_paje_file, "13	%.9f	%smm%u	bw	%f\n",
				itor->comm_start, prefix, itor->src_node, current_bandwidth_per_node[itor->src_node]);
#endif
		}

		current_bandwidth_per_node[itor->dst_node] +=  itor->bandwidth;
		if (out_paje_file)
		{
#ifdef STARPU_HAVE_POTI
			char dst_memnode_container[STARPU_POTI_STR_LEN];
			memmanager_container_alias(dst_memnode_container, STARPU_POTI_STR_LEN, prefix, itor->dst_node);
			poti_SetVariable(itor->comm_start, dst_memnode_container, "bw", current_bandwidth_per_node[itor->dst_node]);
#else
			fprintf(out_paje_file, "13	%.9f	%smm%u	bw	%f\n",
				itor->comm_start, prefix, itor->dst_node, current_bandwidth_per_node[itor->dst_node]);
#endif
		}
	}
}

static
void _starpu_fxt_parse_new_file(char *filename_in, struct starpu_fxt_options *options)
{
	/* Open the trace file */
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0)
	{
	        perror("open failed :");
	        exit(-1);
	}

	static fxt_t fut;
	fut = fxt_fdopen(fd_in);
	if (!fut)
	{
	        perror("fxt_fdopen :");
	        exit(-1);
	}

	fxt_blockev_t block;
	block = fxt_blockev_enter(fut);

	_starpu_symbol_name_list_init(&symbol_list);
	_starpu_communication_list_init(&communication_list);

	char *prefix = options->file_prefix;

	/* TODO starttime ...*/
	/* create the "program" container */
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char new_program_container_alias[STARPU_POTI_STR_LEN], new_program_container_name[STARPU_POTI_STR_LEN];
		program_container_alias(new_program_container_alias, STARPU_POTI_STR_LEN, prefix);
		snprintf(new_program_container_name, STARPU_POTI_STR_LEN, "program %s", prefix);
		poti_CreateContainer (0, new_program_container_alias, "P", "MPIroot", new_program_container_name);
		if (!options->no_counter)
		{
			char new_scheduler_container_alias[STARPU_POTI_STR_LEN], new_scheduler_container_name[STARPU_POTI_STR_LEN];
			scheduler_container_alias(new_scheduler_container_alias, STARPU_POTI_STR_LEN, prefix);
			snprintf(new_scheduler_container_name, STARPU_POTI_STR_LEN, "scheduler %s", prefix);
			poti_CreateContainer(0.0, new_scheduler_container_alias, "Sc", new_program_container_alias, new_scheduler_container_name);
			poti_SetVariable(0.0, new_scheduler_container_alias, "nsubmitted", 0.0);
			poti_SetVariable(0.0, new_scheduler_container_alias, "nready", 0.0);
		}
#else
		fprintf(out_paje_file, "7	0.0	%sp	P	MPIroot	%sprogram \n", prefix, prefix);
		/* create a variable with the number of tasks */
		if (!options->no_counter)
		{
			fprintf(out_paje_file, "7	%.9f	%ssched	Sc	%sp	scheduler\n", 0.0, prefix, prefix);
			fprintf(out_paje_file, "13	0.0	%ssched	nsubmitted	0.0\n", prefix);
			fprintf(out_paje_file, "13	0.0	%ssched	nready	0.0\n", prefix);
		}
#endif
	}

	struct fxt_ev_64 ev;
	while(1)
	{
		unsigned i;
		int ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev);
		for (i = ev.nb_params; i < FXT_MAX_PARAMS; i++)
			ev.param[i] = 0;
		if (ret != FXT_EV_OK)
		{
			break;
		}

		switch (ev.code)
		{
			case _STARPU_FUT_WORKER_INIT_START:
				handle_worker_init_start(&ev, options);
				break;

			case _STARPU_FUT_WORKER_INIT_END:
				handle_worker_init_end(&ev, options);
				break;

			case _STARPU_FUT_NEW_MEM_NODE:
				handle_new_mem_node(&ev, options);
				break;

			/* detect when the workers were idling or not */
			case _STARPU_FUT_START_CODELET_BODY:
				handle_start_codelet_body(&ev, options);
				break;
			case _STARPU_FUT_MODEL_NAME:
				handle_model_name(&ev, options);
				break;
			case _STARPU_FUT_CODELET_DATA:
				handle_codelet_data(&ev, options);
				break;
			case _STARPU_FUT_CODELET_DATA_HANDLE:
				handle_codelet_data_handle(&ev, options);
				break;
			case _STARPU_FUT_CODELET_DETAILS:
				handle_codelet_details(&ev, options);
				break;
			case _STARPU_FUT_END_CODELET_BODY:
				handle_end_codelet_body(&ev, options);
				break;

			case _STARPU_FUT_START_EXECUTING:
				handle_start_executing(&ev, options);
				break;
			case _STARPU_FUT_END_EXECUTING:
				handle_end_executing(&ev, options);
				break;

			case _STARPU_FUT_START_CALLBACK:
				handle_start_callback(&ev, options);
				break;
			case _STARPU_FUT_END_CALLBACK:
				handle_end_callback(&ev, options);
				break;

			case _STARPU_FUT_UPDATE_TASK_CNT:
				handle_update_task_cnt(&ev, options);
				break;

			/* monitor stack size */
			case _STARPU_FUT_JOB_PUSH:
				handle_job_push(&ev, options);
				break;
			case _STARPU_FUT_JOB_POP:
				handle_job_pop(&ev, options);
				break;

			case _STARPU_FUT_SCHED_COMPONENT_NEW:
				handle_component_new(&ev, options);
				break;
			case _STARPU_FUT_SCHED_COMPONENT_CONNECT:
				handle_component_connect(&ev, options);
				break;
			case _STARPU_FUT_SCHED_COMPONENT_PUSH:
				handle_component_push(&ev, options);
				break;
			case _STARPU_FUT_SCHED_COMPONENT_PULL:
				handle_component_pull(&ev, options);
				break;

			/* check the memory transfer overhead */
			case _STARPU_FUT_START_FETCH_INPUT:
				handle_worker_status(&ev, options, "Fi");
				break;
			case _STARPU_FUT_START_PUSH_OUTPUT:
				handle_worker_status(&ev, options, "Po");
				break;
			case _STARPU_FUT_START_PROGRESS:
				handle_worker_status(&ev, options, "P");
				break;
			case _STARPU_FUT_START_UNPARTITION:
				handle_worker_status(&ev, options, "U");
				break;
			case _STARPU_FUT_END_FETCH_INPUT:
			case _STARPU_FUT_END_PROGRESS:
			case _STARPU_FUT_END_PUSH_OUTPUT:
			case _STARPU_FUT_END_UNPARTITION:
				handle_worker_status(&ev, options, "B");
				break;

			case _STARPU_FUT_WORKER_SCHEDULING_START:
				handle_worker_scheduling_start(&ev, options);
				break;

			case _STARPU_FUT_WORKER_SCHEDULING_END:
				handle_worker_scheduling_end(&ev, options);
				break;

			case _STARPU_FUT_WORKER_SCHEDULING_PUSH:
				handle_worker_scheduling_push(&ev, options);
				break;

			case _STARPU_FUT_WORKER_SCHEDULING_POP:
				handle_worker_scheduling_pop(&ev, options);
				break;

			case _STARPU_FUT_WORKER_SLEEP_START:
				handle_worker_sleep_start(&ev, options);
				break;

			case _STARPU_FUT_WORKER_SLEEP_END:
				handle_worker_sleep_end(&ev, options);
				break;

			case _STARPU_FUT_TAG:
				handle_tag(&ev);
				break;

			case _STARPU_FUT_TAG_DEPS:
				handle_tag_deps(&ev);
				break;

			case _STARPU_FUT_TASK_DEPS:
				handle_task_deps(&ev);
				break;

			case _STARPU_FUT_TASK_SUBMIT:
				handle_task_submit(&ev, options);
				break;

			case _STARPU_FUT_TASK_DONE:
				handle_task_done(&ev, options);
				break;

			case _STARPU_FUT_TAG_DONE:
				handle_tag_done(&ev, options);
				break;

			case _STARPU_FUT_DATA_COPY:
				if (!options->no_bus)
				     handle_data_copy();
				break;

			case _STARPU_FUT_DATA_LOAD:
			     	break;

			case _STARPU_FUT_START_DRIVER_COPY:
				if (!options->no_bus)
					handle_start_driver_copy(&ev, options);
				break;

			case _STARPU_FUT_END_DRIVER_COPY:
				if (!options->no_bus)
					handle_end_driver_copy(&ev, options);
				break;

			case _STARPU_FUT_START_DRIVER_COPY_ASYNC:
				if (!options->no_bus)
					handle_start_driver_copy_async(&ev, options);
				break;

			case _STARPU_FUT_END_DRIVER_COPY_ASYNC:
				if (!options->no_bus)
					handle_end_driver_copy_async(&ev, options);
				break;

			case _STARPU_FUT_WORK_STEALING:
				handle_work_stealing(&ev, options);
				break;

			case _STARPU_FUT_WORKER_DEINIT_START:
				handle_worker_deinit_start(&ev, options);
				break;

			case _STARPU_FUT_WORKER_DEINIT_END:
				handle_worker_deinit_end(&ev, options);
				break;

			case _STARPU_FUT_START_ALLOC:
				if (!options->no_bus)
					handle_memnode_event(&ev, options, "A");
				break;
			case _STARPU_FUT_START_ALLOC_REUSE:
				if (!options->no_bus)
					handle_memnode_event(&ev, options, "Ar");
				break;
			case _STARPU_FUT_END_ALLOC:
			case _STARPU_FUT_END_ALLOC_REUSE:
				if (!options->no_bus)
				handle_memnode_event(&ev, options, "No");
				break;
			case _STARPU_FUT_START_FREE:
				if (!options->no_bus)
				{
					handle_memnode_event(&ev, options, "F");
				}
				break;
			case _STARPU_FUT_END_FREE:
				if (!options->no_bus)
				{
					unsigned memnode = ev.param[0];
					if (reclaiming[memnode])
						handle_memnode_event(&ev, options, "R");
					else
						handle_memnode_event(&ev, options, "No");
				}
				break;
			case _STARPU_FUT_START_WRITEBACK:
				if (!options->no_bus)
				{
					handle_memnode_event(&ev, options, "W");
				}
				break;
			case _STARPU_FUT_END_WRITEBACK:
				if (!options->no_bus)
				{
					unsigned memnode = ev.param[0];
					if (reclaiming[memnode])
						handle_memnode_event(&ev, options, "R");
					else
						handle_memnode_event(&ev, options, "No");
				}
				break;
			case _STARPU_FUT_START_WRITEBACK_ASYNC:
				if (!options->no_bus)
				{
					handle_memnode_event(&ev, options, "Wa");
				}
				break;
			case _STARPU_FUT_END_WRITEBACK_ASYNC:
				if (!options->no_bus)
				{
					unsigned memnode = ev.param[0];
					if (reclaiming[memnode])
						handle_memnode_event(&ev, options, "R");
					else
						handle_memnode_event(&ev, options, "No");
				}
				break;
			case _STARPU_FUT_START_MEMRECLAIM:
				if (!options->no_bus)
				{
					unsigned memnode = ev.param[0];
					reclaiming[memnode] = 1;
					handle_memnode_event(&ev, options, "R");
				}
				break;
			case _STARPU_FUT_END_MEMRECLAIM:
				if (!options->no_bus)
				{
					unsigned memnode = ev.param[0];
					reclaiming[memnode] = 0;
					handle_memnode_event(&ev, options, "No");
				}
				break;

			case _STARPU_FUT_USER_EVENT:
				handle_user_event(&ev, options);
				break;

			case _STARPU_MPI_FUT_START:
				handle_mpi_start(&ev, options);
				break;

			case _STARPU_MPI_FUT_STOP:
				handle_mpi_stop(&ev, options);
				break;

			case _STARPU_MPI_FUT_BARRIER:
				handle_mpi_barrier(&ev, options);
				break;

			case _STARPU_MPI_FUT_ISEND_SUBMIT_BEGIN:
				handle_mpi_isend_submit_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_ISEND_SUBMIT_END:
				handle_mpi_isend_submit_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_IRECV_SUBMIT_BEGIN:
				handle_mpi_irecv_submit_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_IRECV_SUBMIT_END:
				handle_mpi_irecv_submit_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_ISEND_COMPLETE_BEGIN:
				handle_mpi_isend_complete_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_ISEND_COMPLETE_END:
				handle_mpi_isend_complete_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_IRECV_COMPLETE_BEGIN:
				handle_mpi_irecv_complete_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_IRECV_COMPLETE_END:
				handle_mpi_irecv_complete_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_SLEEP_BEGIN:
				handle_mpi_sleep_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_SLEEP_END:
				handle_mpi_sleep_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_DTESTING_BEGIN:
				handle_mpi_dtesting_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_DTESTING_END:
				handle_mpi_dtesting_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_UTESTING_BEGIN:
				handle_mpi_utesting_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_UTESTING_END:
				handle_mpi_utesting_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_UWAIT_BEGIN:
				handle_mpi_uwait_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_UWAIT_END:
				handle_mpi_uwait_end(&ev, options);
				break;

			case _STARPU_FUT_SET_PROFILING:
				handle_set_profiling(&ev, options);
				break;

			case _STARPU_FUT_TASK_WAIT_FOR_ALL:
				handle_task_wait_for_all();
				break;

			case _STARPU_FUT_EVENT:
				handle_event(&ev, options);
				break;

			case _STARPU_FUT_THREAD_EVENT:
				handle_thread_event(&ev, options);
				break;

			case _STARPU_FUT_LOCKING_MUTEX:
				break;

			case _STARPU_FUT_MUTEX_LOCKED:
				break;

			case _STARPU_FUT_UNLOCKING_MUTEX:
				break;

			case _STARPU_FUT_MUTEX_UNLOCKED:
				break;

			case _STARPU_FUT_TRYLOCK_MUTEX:
				break;

			case _STARPU_FUT_RDLOCKING_RWLOCK:
				break;

			case _STARPU_FUT_RWLOCK_RDLOCKED:
				break;

			case _STARPU_FUT_WRLOCKING_RWLOCK:
				break;

			case _STARPU_FUT_RWLOCK_WRLOCKED:
				break;

			case _STARPU_FUT_UNLOCKING_RWLOCK:
				break;

			case _STARPU_FUT_RWLOCK_UNLOCKED:
				break;

			case _STARPU_FUT_LOCKING_SPINLOCK:
				break;

			case _STARPU_FUT_SPINLOCK_LOCKED:
				break;

			case _STARPU_FUT_UNLOCKING_SPINLOCK:
				break;

			case _STARPU_FUT_SPINLOCK_UNLOCKED:
				break;

			case _STARPU_FUT_TRYLOCK_SPINLOCK:
				break;

			case _STARPU_FUT_COND_WAIT_BEGIN:
				break;

			case _STARPU_FUT_COND_WAIT_END:
				break;

			case _STARPU_FUT_BARRIER_WAIT_BEGIN:
				break;

			case _STARPU_FUT_BARRIER_WAIT_END:
				break;

			case _STARPU_FUT_MEMORY_FULL:
				break;

			case _STARPU_FUT_SCHED_COMPONENT_POP_PRIO:
				break;

			case _STARPU_FUT_SCHED_COMPONENT_PUSH_PRIO:
				break;

			case _STARPU_FUT_HYPERVISOR_BEGIN:
				handle_hypervisor_begin(&ev, options);
				break;

			case _STARPU_FUT_HYPERVISOR_END:
				handle_hypervisor_end(&ev, options);
				break;

			/* We can safely ignore FUT internal events */
			case FUT_SETUP_CODE:
			case FUT_CALIBRATE0_CODE:
			case FUT_CALIBRATE1_CODE:
			case FUT_CALIBRATE2_CODE:
			case FUT_KEYCHANGE_CODE:
			case FUT_NEW_LWP_CODE:
			case FUT_GCC_INSTRUMENT_ENTRY_CODE:
				break;

			default:
#ifdef STARPU_VERBOSE
				fprintf(stderr, "unknown event.. %x at time %llx WITH OFFSET %llx\n",
					(unsigned)ev.code, (long long unsigned)ev.time, (long long unsigned)(ev.time-options->file_offset));
#endif
				break;
		}
	}

	/* Close the trace file */
	if (close(fd_in))
	{
	        perror("close failed :");
	        exit(-1);
	}
}

/* Initialize FxT options to default values */
void starpu_fxt_options_init(struct starpu_fxt_options *options)
{
	options->per_task_colour = 0;
	options->no_counter = 0;
	options->no_bus = 0;
	options->ninputfiles = 0;
	options->out_paje_path = "paje.trace";
	options->dag_path = "dag.dot";
	options->tasks_path = "tasks.rec";
	options->anim_path = "trace.html";
	options->distrib_time_path = "distrib.data";
	options->dumped_codelets = NULL;
	options->activity_path = "activity.data";
}

static
void _starpu_fxt_distrib_file_init(struct starpu_fxt_options *options)
{
	dumped_codelets_count = 0;
	dumped_codelets = NULL;

	if (options->distrib_time_path)
	{
		distrib_time = fopen(options->distrib_time_path, "w+");
	}
	else
	{
		distrib_time = NULL;
	}
}

static
void _starpu_fxt_distrib_file_close(struct starpu_fxt_options *options)
{
	if (distrib_time)
		fclose(distrib_time);

	if (options->dumped_codelets)
	{
		*options->dumped_codelets = dumped_codelets;
		options->dumped_codelets_count = dumped_codelets_count;
	}
}

static
void _starpu_fxt_activity_file_init(struct starpu_fxt_options *options)
{
	if (options->activity_path)
		activity_file = fopen(options->activity_path, "w+");
	else
		activity_file = NULL;
}

static
void _starpu_fxt_anim_file_init(struct starpu_fxt_options *options)
{
	if (options->anim_path)
	{
		anim_file = fopen(options->anim_path, "w+");
		_starpu_fxt_component_print_header(anim_file);
	}
	else
		anim_file = NULL;
}

static
void _starpu_fxt_tasks_file_init(struct starpu_fxt_options *options)
{
	if (options->tasks_path)
		tasks_file = fopen(options->tasks_path, "w+");
	else
		tasks_file = NULL;
}

static
void _starpu_fxt_activity_file_close(void)
{
	if (activity_file)
		fclose(activity_file);
}

static
void _starpu_fxt_anim_file_close(void)
{
	//_starpu_fxt_component_dump(stderr);
	_starpu_fxt_component_finish(anim_file);
	if (anim_file)
		fclose(anim_file);
}

static
void _starpu_fxt_tasks_file_close(void)
{
	if (tasks_file)
		fclose(tasks_file);
}

static
void _starpu_fxt_paje_file_init(struct starpu_fxt_options *options)
{
	/* create a new file */
	if (options->out_paje_path)
	{
		out_paje_file = fopen(options->out_paje_path, "w+");
		if (!out_paje_file)
		{
			fprintf(stderr,"error while opening %s\n", options->out_paje_path);
			perror("fopen");
			exit(1);
		}

#ifdef STARPU_HAVE_POTI
		poti_init (out_paje_file);
#endif
		_starpu_fxt_write_paje_header(out_paje_file);
	}
	else
	{
		out_paje_file = NULL;
	}
}

static
void _starpu_fxt_paje_file_close(void)
{
	if (out_paje_file)
		fclose(out_paje_file);
}

static
uint64_t _starpu_fxt_find_start_time(char *filename_in)
{
	/* Open the trace file */
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0)
	{
	        perror("open failed :");
	        exit(-1);
	}

	static fxt_t fut;
	fut = fxt_fdopen(fd_in);
	if (!fut)
	{
	        perror("fxt_fdopen :");
	        exit(-1);
	}

	fxt_blockev_t block;
	block = fxt_blockev_enter(fut);

	struct fxt_ev_64 ev;

	int ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev);
	STARPU_ASSERT (ret == FXT_EV_OK);

	/* Close the trace file */
	if (close(fd_in))
	{
	        perror("close failed :");
	        exit(-1);
	}
	return (ev.time);
}

void starpu_fxt_generate_trace(struct starpu_fxt_options *options)
{
	_starpu_fxt_dag_init(options->dag_path);
	_starpu_fxt_distrib_file_init(options);
	_starpu_fxt_activity_file_init(options);
	_starpu_fxt_anim_file_init(options);
	_starpu_fxt_tasks_file_init(options);

	_starpu_fxt_paje_file_init(options);

	if (options->ninputfiles == 0)
	{
		return;
	}
	else if (options->ninputfiles == 1)
	{
		/* we usually only have a single trace */
		uint64_t file_start_time = _starpu_fxt_find_start_time(options->filenames[0]);
		options->file_prefix = "";
		options->file_offset = file_start_time;
		options->file_rank = -1;

		_starpu_fxt_parse_new_file(options->filenames[0], options);
	}
	else
	{
		unsigned inputfile;

		uint64_t offsets[options->ninputfiles];

		/*
		 * Find the trace offsets:
		 *	- If there is no sync point
		 *		psi_k(x) = x - start_k
		 *	- If there is a sync point sync_k
		 *		psi_k(x) = x - sync_k + M
		 *		where M = max { sync_i - start_i | there exists sync_i}
		 * More generally:
		 *	- psi_k(x) = x - offset_k
		 */

		int unique_keys[options->ninputfiles];
		int rank_k[options->ninputfiles];
		uint64_t start_k[options->ninputfiles];
		uint64_t sync_k[options->ninputfiles];
		unsigned sync_k_exists[options->ninputfiles];
		uint64_t M = 0;

		unsigned found_one_sync_point = 0;
		int key = 0;
		unsigned display_mpi = 0;

		/* Compute all start_k */
		for (inputfile = 0; inputfile < options->ninputfiles; inputfile++)
		{
			uint64_t file_start = _starpu_fxt_find_start_time(options->filenames[inputfile]);
			start_k[inputfile] = file_start;
		}

		/* Compute all sync_k if they exist */
		for (inputfile = 0; inputfile < options->ninputfiles; inputfile++)
		{
			int ret = _starpu_fxt_mpi_find_sync_point(options->filenames[inputfile],
								  &sync_k[inputfile],
								  &unique_keys[inputfile],
								  &rank_k[inputfile]);
			if (ret == -1)
			{
				/* There was no sync point, we assume there is no offset */
				sync_k_exists[inputfile] = 0;
			}
			else
			{
				if (!found_one_sync_point)
				{
					key = unique_keys[inputfile];
					display_mpi = 1;
					found_one_sync_point = 1;
				}
				else
				{
					if (key != unique_keys[inputfile])
					{
						fprintf(stderr, "Warning: traces are coming from different run so we will not try to display MPI communications.\n");
						display_mpi = 0;
					}
				}


				STARPU_ASSERT(sync_k[inputfile] >= start_k[inputfile]);

				sync_k_exists[inputfile] = 1;

				uint64_t diff = sync_k[inputfile] - start_k[inputfile];
				if (diff > M)
					M = diff;
			}
		}

		/* Compute the offset */
		for (inputfile = 0; inputfile < options->ninputfiles; inputfile++)
		{
			offsets[inputfile] = sync_k_exists[inputfile]?
						(sync_k[inputfile]-M):start_k[inputfile];
		}

		/* generate the Paje trace for the different files */
		for (inputfile = 0; inputfile < options->ninputfiles; inputfile++)
		{
			int filerank = rank_k[inputfile];

			_STARPU_DISP("Parsing file %s (rank %d)\n", options->filenames[inputfile], filerank);

			char file_prefix[32];
			snprintf(file_prefix, sizeof(file_prefix), "%d_", filerank);

			options->file_prefix = file_prefix;
			options->file_offset = offsets[inputfile];
			options->file_rank = filerank;

			_starpu_fxt_parse_new_file(options->filenames[inputfile], options);
		}

		/* display the MPI transfers if possible */
		if (display_mpi)
			_starpu_fxt_display_mpi_transfers(options, rank_k, out_paje_file);
	}

	_starpu_fxt_display_bandwidth(options);

	/* close the different files */
	_starpu_fxt_paje_file_close();
	_starpu_fxt_activity_file_close();
	_starpu_fxt_distrib_file_close(options);
	_starpu_fxt_anim_file_close();
	_starpu_fxt_tasks_file_close();

	_starpu_fxt_dag_terminate();

	options->nworkers = nworkers;
}

#define DATA_STR_MAX_SIZE 15

struct parse_task
{
	unsigned exec_time;
	unsigned data_total;
	char *codelet_name;
};

static struct parse_task tasks[STARPU_NMAXWORKERS];

struct starpu_data_trace_kernel
{
	UT_hash_handle hh;
	char *name;
	FILE *file;
} *kernels;

#define NANO_SEC_TO_MILI_SEC 0.000001

static FILE *codelet_list;

static void write_task(struct parse_task pt)
{
	struct starpu_data_trace_kernel *kernel;
	char *codelet_name = pt.codelet_name;
	HASH_FIND_STR(kernels, codelet_name, kernel);
	//fprintf(stderr, "%p %p %s\n", kernel, kernels, codelet_name);
	if(kernel == NULL)
	{
		kernel = malloc(sizeof(*kernel));
		kernel->name = strdup(codelet_name);
		//fprintf(stderr, "%s\n", kernel->name);
		kernel->file = fopen(codelet_name, "w+");
		if(!kernel->file)
		{
			perror("open failed :");
			exit(-1);
		}
		HASH_ADD_STR(kernels, name, kernel);
		fprintf(codelet_list, "%s\n", codelet_name);
	}
	double time = pt.exec_time * NANO_SEC_TO_MILI_SEC;
	fprintf(kernel->file, "%lf %d\n", time, pt.data_total);
}

void starpu_fxt_write_data_trace(char *filename_in)
{
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0)
	{
	        perror("open failed :");
	        exit(-1);
	}

	static fxt_t fut;
	fut = fxt_fdopen(fd_in);
	if (!fut)
	{
	        perror("fxt_fdopen :");
	        exit(-1);
	}

	codelet_list = fopen("codelet_list", "w+");
	if(!codelet_list)
	{
		perror("open failed :");
		exit(-1);
	}

	fxt_blockev_t block;
	block = fxt_blockev_enter(fut);

	struct fxt_ev_64 ev;

	int workerid=-1;
	unsigned long has_name = 0;

	while(1)
	{
		int ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev);
		if (ret != FXT_EV_OK)
		{
			break;
		}

		switch (ev.code)
		{
		case _STARPU_FUT_WORKER_INIT_START:
			register_worker_id(ev.param[6], ev.param[1], ev.param[5]);
			break;

		case _STARPU_FUT_START_CODELET_BODY:
			workerid = ev.param[2];
			tasks[workerid].exec_time = ev.time;
			has_name = ev.param[3];
			tasks[workerid].codelet_name = strdup(has_name ? (char *) &ev.param[4] : "unknown");
			//fprintf(stderr, "start codelet :[%d][%s]\n", workerid, tasks[workerid].codelet_name);
			break;

		case _STARPU_FUT_END_CODELET_BODY:
			workerid = ev.param[3];
			assert(workerid != -1);
			tasks[workerid].exec_time = ev.time - tasks[workerid].exec_time;
			write_task(tasks[workerid]);
			break;

		case _STARPU_FUT_DATA_LOAD:
			workerid = ev.param[0];
			tasks[workerid].data_total = ev.param[1];
			break;

		default:
#ifdef STARPU_VERBOSE
			fprintf(stderr, "unknown event.. %x at time %llx WITH OFFSET %llx\n",
				(unsigned)ev.code, (long long unsigned)ev.time, (long long unsigned)(ev.time));
#endif
			break;
		}
	}

	if (close(fd_in))
	{
	        perror("close failed :");
	        exit(-1);
	}

	if(fclose(codelet_list))
	{
		perror("close failed :");
		exit(-1);
	}

	struct starpu_data_trace_kernel *kernel, *tmp;

	HASH_ITER(hh, kernels, kernel, tmp)
	{
		if(fclose(kernel->file))
		{
			perror("close failed :");
			exit(-1);
		}
		HASH_DEL(kernels, kernel);
		free(kernel->name);
		free(kernel);
	}

}
#endif // STARPU_USE_FXT
