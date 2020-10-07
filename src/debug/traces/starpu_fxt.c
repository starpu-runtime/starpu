/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Joris Pablo
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2017-2020  Federal University of Rio Grande do Sul (UFRGS)
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
#include <datawizard/copy_driver.h>
#include <string.h>

#ifdef STARPU_HAVE_POTI
#include <poti.h>
#define STARPU_POTI_STR_LEN 200
#endif
#define STARPU_TRACE_STR_LEN 200

#ifdef STARPU_USE_FXT
#include "starpu_fxt.h"
#include <inttypes.h>
#include <starpu_hash.h>

#define CPUS_WORKER_COLORS_NB	8
#define CUDA_WORKER_COLORS_NB	9
#define OPENCL_WORKER_COLORS_NB 9
#define MIC_WORKER_COLORS_NB	9
#define MPI_MS_WORKER_COLORS_NB	9
#define OTHER_WORKER_COLORS_NB	4

/* How many times longer an idle period has to be before the smoothing
 * heuristics avoids averaging codelet gflops */
#define IDLE_FACTOR 2

static char *cpus_worker_colors[CPUS_WORKER_COLORS_NB] = {"/greens9/7", "/greens9/6", "/greens9/5", "/greens9/4",  "/greens9/9", "/greens9/3",  "/greens9/2",  "/greens9/1"  };
static char *cuda_worker_colors[CUDA_WORKER_COLORS_NB] = {"/ylorrd9/9", "/ylorrd9/6", "/ylorrd9/3", "/ylorrd9/1", "/ylorrd9/8", "/ylorrd9/7", "/ylorrd9/4", "/ylorrd9/2",  "/ylorrd9/1"};
static char *opencl_worker_colors[OPENCL_WORKER_COLORS_NB] = {"/blues9/9", "/blues9/6", "/blues9/3", "/blues9/1", "/blues9/8", "/blues9/7", "/blues9/4", "/blues9/2",  "/blues9/1"};
static char *mic_worker_colors[MIC_WORKER_COLORS_NB] = {"/reds9/9", "/reds9/6", "/reds9/3", "/reds9/1", "/reds9/8", "/reds9/7", "/reds9/4", "/reds9/2",  "/reds9/1"};
static char *mpi_ms_worker_colors[MPI_MS_WORKER_COLORS_NB] = {"/reds9/9", "/reds9/6", "/reds9/3", "/reds9/1", "/reds9/8", "/reds9/7", "/reds9/4", "/reds9/2",  "/reds9/1"};
static char *other_worker_colors[OTHER_WORKER_COLORS_NB] = {"/greys9/9", "/greys9/8", "/greys9/7", "/greys9/6"};
static char *worker_colors[STARPU_NMAXWORKERS];

static unsigned opencl_index = 0;
static unsigned cuda_index = 0;
static unsigned cpus_index = 0;
static unsigned mic_index = 0;
static unsigned mpi_ms_index = 0;
static unsigned other_index = 0;

static unsigned long fut_keymask;

/* Get pointer to string starting at nth parameter */
static char *get_fxt_string(struct fxt_ev_64 *ev, int n)
{
	char *s = (char *)&ev->param[n];
	s[(FXT_MAX_PARAMS-n)*sizeof(unsigned long) - 1] = 0;
	return s;
}

/*
 * Paje trace file tools
 */

static FILE *out_paje_file;
static FILE *distrib_time;
static FILE *activity_file;
static FILE *anim_file;
static FILE *tasks_file;
static FILE *data_file;
static FILE *trace_file;

struct data_parameter_info
{
	unsigned long handle;
	unsigned long size;
	int mode;
};

struct task_info
{
	UT_hash_handle hh;
	char *model_name;
	char *name;
	int exclude_from_dag;
	int show;
	unsigned type;
	unsigned long job_id;
	unsigned long submit_order;
	long priority;
	int color;
	uint64_t tag;
	int workerid;
	int node;
	double submit_time;
	double start_time;
	double end_time;
	unsigned long footprint;
	unsigned long kflops;
	long iterations[2];
	char *parameters;
	unsigned int ndeps;
	unsigned long *dependencies;
	char **dep_labels;
	unsigned long ndata;
	struct data_parameter_info *data;
	int mpi_rank;
};

struct task_info *tasks_info;

static struct task_info *get_task(unsigned long job_id, int mpi_rank)
{
	struct task_info *task;

	HASH_FIND(hh, tasks_info, &job_id, sizeof(job_id), task);
	if (!task)
	{
		unsigned i;
		_STARPU_MALLOC(task, sizeof(*task));
		task->model_name = NULL;
		task->name = NULL;
		task->exclude_from_dag = 0;
		task->show = 0;
		task->type = 0;
		task->job_id = job_id;
		task->submit_order = 0;
		task->priority = 0;
		task->color = 0;
		task->tag = 0;
		task->workerid = -1;
		task->node = -1;
		task->submit_time = 0.;
		task->start_time = 0.;
		task->end_time = 0.;
		task->footprint = 0;
		task->kflops = 0.;
		for (i = 0; i < sizeof(task->iterations)/sizeof(task->iterations[0]); i++)
			task->iterations[i] = -1;
		task->parameters = NULL;
		task->ndeps = 0;
		task->dependencies = NULL;
		task->dep_labels = NULL;
		task->ndata = 0;
		task->data = NULL;
		task->mpi_rank = mpi_rank;
		HASH_ADD(hh, tasks_info, job_id, sizeof(task->job_id), task);
	}
	else
		STARPU_ASSERT(task->mpi_rank == mpi_rank);

	return task;
}

/* Return whether to show this task in the DAG or not */
static int show_task(struct task_info *task, struct starpu_fxt_options *options)
{
	if (task->show)
		return 1;
	if (task->type & STARPU_TASK_TYPE_INTERNAL && !options->internal)
		return 0;
	if (task->type & STARPU_TASK_TYPE_DATA_ACQUIRE && options->no_acquire)
		return 0;
	return 1;
}

static void task_dump(struct task_info *task, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	unsigned i;

	if (task->exclude_from_dag)
		goto out;
	if (!tasks_file)
		goto out;

	if (task->name)
	{
		fprintf(tasks_file, "Name: %s\n", task->name);
		free(task->name);
	}
	if (task->model_name)
	{
		fprintf(tasks_file, "Model: %s\n", task->model_name);
		free(task->model_name);
	}
	fprintf(tasks_file, "JobId: %s%lu\n", prefix, task->job_id);
	if (task->submit_order)
		fprintf(tasks_file, "SubmitOrder: %lu\n", task->submit_order);
	fprintf(tasks_file, "Priority: %ld\n", task->priority);
	if (task->dependencies)
	{
		fprintf(tasks_file, "DependsOn:");
		for (i = 0; i < task->ndeps; i++)
			fprintf(tasks_file, " %s%lu", prefix, task->dependencies[i]);
		fprintf(tasks_file, "\n");
		free(task->dependencies);
	}
	if (task->dep_labels)
	{
		fprintf(tasks_file, "DepLabels:");
		for (i = 0; i < task->ndeps; i++)
		{
			fprintf(tasks_file, " %s", task->dep_labels[i]);
			free(task->dep_labels[i]);
		}
		fprintf(tasks_file, "\n");
		free(task->dep_labels);
	}
	fprintf(tasks_file, "Tag: %"PRIx64"\n", task->tag);
	if (task->workerid >= 0)
		fprintf(tasks_file, "WorkerId: %d\n", task->workerid);
	if (task->node >= 0)
		fprintf(tasks_file, "MemoryNode: %d\n", task->node);
	if (task->submit_time != 0.)
		fprintf(tasks_file, "SubmitTime: %f\n", task->submit_time);
	if (task->start_time != 0.)
		fprintf(tasks_file, "StartTime: %f\n", task->start_time);
	if (task->end_time != 0.)
		fprintf(tasks_file, "EndTime: %f\n", task->end_time);
	fprintf(tasks_file, "Footprint: %lx\n", task->footprint);
	if (task->kflops != 0)
		fprintf(tasks_file, "GFlop: %f\n", ((double) task->kflops) / 1000000);
	if (task->iterations[0] != -1)
	{
		fprintf(tasks_file, "Iteration:");
		for (i = 0; i < sizeof(task->iterations)/sizeof(task->iterations[0]); i++)
		{
			if (task->iterations[i] == -1)
				break;
			fprintf(tasks_file, " %ld", task->iterations[i]);
		}
		fprintf(tasks_file, "\n");
	}
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
	fprintf(tasks_file, "MPIRank: %d\n", task->mpi_rank);
	fprintf(tasks_file, "\n");

out:
	HASH_DEL(tasks_info, task);
	free(task);
}

struct data_info
{
	UT_hash_handle hh;
	unsigned long handle;
	char *name;
	size_t size;
	char *description;
	unsigned dimensions;
	unsigned long *dims;
	int home_node;
	int mpi_rank;
	int mpi_owner;
	long mpi_tag;
};

struct data_info *data_info;

static struct data_info *get_data(unsigned long handle, int mpi_rank)
{
	struct data_info *data;

	HASH_FIND(hh, data_info, &handle, sizeof(handle), data);
	if (!data)
	{
		_STARPU_MALLOC(data, sizeof(*data));
		data->handle = handle;
		data->name = NULL;
		data->size = 0;
		data->description = 0;
		data->dimensions = 0;
		data->dims = NULL;
		data->home_node = STARPU_MAIN_RAM;
		data->mpi_rank = mpi_rank;
		data->mpi_owner = mpi_rank;
		data->mpi_tag = -1;
		HASH_ADD(hh, data_info, handle, sizeof(handle), data);
	}
	else
		STARPU_ASSERT(data->mpi_rank == mpi_rank);

	return data;
}

static void data_dump(struct data_info *data)
{
	if (!data_file)
		goto out;
	fprintf(data_file, "Handle: %lx\n", data->handle);
	fprintf(data_file, "HomeNode: %d\n", data->home_node);
	if (data->mpi_rank >= 0)
		fprintf(data_file, "MPIRank: %d\n", data->mpi_rank);
	if (data->name)
	{
		fprintf(data_file, "Name: %s\n", data->name);
		free(data->name);
	}
	fprintf(data_file, "Size: %lu\n", (unsigned long) data->size);
	if (data->description)
	{
		fprintf(data_file, "Description: %s\n", data->description);
		free(data->description);
	}
	if (data->dimensions)
	{
		unsigned i;
		fprintf(data_file, "Coordinates:");
		for (i = 0; i < data->dimensions; i++)
			fprintf(data_file, " %lu", data->dims[i]);
		fprintf(data_file, "\n");
	}
	if (data->mpi_owner >= 0)
		fprintf(data_file, "MPIOwner: %d\n", data->mpi_owner);
	if (data->mpi_tag >= 0)
		fprintf(data_file, "MPITag: %ld\n", data->mpi_tag);
	fprintf(data_file, "\n");
out:
	HASH_DEL(data_info, data);
	free(data);
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

static void set_next_mpi_ms_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		return;
	worker_colors[workerid] = mpi_ms_worker_colors[mpi_ms_index++];
	if (mpi_ms_index == MPI_MS_WORKER_COLORS_NB) mpi_ms_index = 0;
}

static const char *get_worker_color(int workerid)
{
	if (workerid >= STARPU_NMAXWORKERS)
		workerid = STARPU_NMAXWORKERS - 1;
	return worker_colors[workerid];
}

static unsigned get_color_symbol_red(char *name)
{
	/* choose some color ... that's disguting yes */
	uint32_t hash_symbol = starpu_hash_crc32c_string(name, 0);
	return (unsigned)starpu_hash_crc32c_string("red", hash_symbol) % 1024;
}

static unsigned get_color_symbol_green(char *name)
{
	/* choose some color ... that's disguting yes */
	uint32_t hash_symbol = starpu_hash_crc32c_string(name, 0);
	return (unsigned)starpu_hash_crc32c_string("green", hash_symbol) % 1024;
}

static unsigned get_color_symbol_blue(char *name)
{
	/* choose some color ... that's disguting yes */
	uint32_t hash_symbol = starpu_hash_crc32c_string(name, 0);
	return (unsigned)starpu_hash_crc32c_string("blue", hash_symbol) % 1024;
}

/* Start time of last codelet for this worker */
static double last_codelet_start[STARPU_NMAXWORKERS];
/* End time of last codelet for this worker */
static double last_codelet_end[STARPU_NMAXWORKERS];
/* _STARPU_FUT_DO_PROBE5STR records only 3 longs */
char _starpu_last_codelet_symbol[STARPU_NMAXWORKERS][(FXT_MAX_PARAMS-5)*sizeof(unsigned long)];
static int last_codelet_parameter[STARPU_NMAXWORKERS];
#define MAX_PARAMETERS 8
static char last_codelet_parameter_description[STARPU_NMAXWORKERS][MAX_PARAMETERS][FXT_MAX_PARAMS*sizeof(unsigned long)];

/* If more than a period of time has elapsed, we flush the profiling info,
 * otherwise they are accumulated everytime there is a new relevant event. */
#define ACTIVITY_PERIOD	75.0
static double last_activity_flush_timestamp[STARPU_NMAXWORKERS];
static double accumulated_sleep_time[STARPU_NMAXWORKERS];
static double accumulated_exec_time[STARPU_NMAXWORKERS];

static unsigned steal_number = 0;

LIST_TYPE(_starpu_symbol_name,
	char *name;
)

static struct _starpu_symbol_name_list symbol_list;

/* List of on-going communications */
LIST_TYPE(_starpu_communication,
	unsigned comid;
	double comm_start;
	double bandwidth;
	unsigned src_node;
	unsigned dst_node;
	unsigned long size;
	const char *type;
	unsigned long handle;
	struct _starpu_communication *peer;
)

static struct _starpu_communication_list communication_list;
static double current_bandwidth_in_per_node[STARPU_MAXNODES] = {0.0};
static double current_bandwidth_out_per_node[STARPU_MAXNODES] = {0.0};

/* List of on-going computations */
LIST_TYPE(_starpu_computation,
	double comp_start;
	double gflops;
	struct _starpu_computation *peer;
)

/* List of ongoing computations */
static struct _starpu_computation_list computation_list;
/* Last computation for each worker */
static struct _starpu_computation *ongoing_computation[STARPU_NMAXWORKERS];

/* Current total GFlops */
static double current_computation;
/* Time of last update of current total GFlops */
static double current_computation_time;

/*
 * Generic tools
 */

#define WORKER_STATE      (1 << 0)
#define THREAD_STATE      (1 << 1)
#define COMM_THREAD_STATE (1 << 2)
#define USER_THREAD_STATE (1 << 3)

static struct
{
	const char *short_name;
	const char *long_name;
	uint8_t flags;
} states_list[] =
{
	{ "Fi",  "FetchingInput",		 WORKER_STATE | THREAD_STATE },
	{ "Po",	 "PushingOutput",		 WORKER_STATE | THREAD_STATE },
	{ "P",	 "Progressing",			 WORKER_STATE | THREAD_STATE },
	{ "U",	 "Unpartitioning",		 WORKER_STATE | THREAD_STATE },
	{ "B",	 "Overhead",			 WORKER_STATE | THREAD_STATE },
	{ "In",	 "Initializing",		 WORKER_STATE | THREAD_STATE },
	{ "D",	 "Deinitializing",		 WORKER_STATE | THREAD_STATE },
	{ "E",	 "Executing",			 WORKER_STATE | THREAD_STATE },
	{ "C",	 "Callback",			 WORKER_STATE | THREAD_STATE | USER_THREAD_STATE },
	{ "H",	 "Hypervisor",			 WORKER_STATE | THREAD_STATE },
	{ "Sc",	 "Scheduling",			 WORKER_STATE | THREAD_STATE },
	{ "I",	 "Idle",			 WORKER_STATE | THREAD_STATE },
	{ "Sl",	 "Sleeping",			 WORKER_STATE | THREAD_STATE | COMM_THREAD_STATE },
	{ "Bu",	 "Building task",		 THREAD_STATE | COMM_THREAD_STATE | USER_THREAD_STATE },
	{ "Su",  "Submitting task",		 THREAD_STATE | COMM_THREAD_STATE | USER_THREAD_STATE },
	{ "Th",  "Throttling task submission",	 THREAD_STATE | COMM_THREAD_STATE | USER_THREAD_STATE },
	{ "MD",  "Decoding task for MPI",	 THREAD_STATE | USER_THREAD_STATE },
	{ "MPr", "Preparing task for MPI",	 THREAD_STATE | USER_THREAD_STATE },
	{ "MPo", "Post-processing task for MPI", THREAD_STATE | USER_THREAD_STATE },
	{ "P",	 "Processing",			 COMM_THREAD_STATE },
	{ "UT",	 "UserTesting",			 COMM_THREAD_STATE },
	{ "UW",	 "UserWaiting",			 COMM_THREAD_STATE },
	{ "SdS", "SendSubmitted",		 COMM_THREAD_STATE },
	{ "RvS", "ReceiveSubmitted",		 COMM_THREAD_STATE },
	{ "SdC", "SendCompleted",		 COMM_THREAD_STATE },
	{ "RvC", "ReceiveCompleted",		 COMM_THREAD_STATE },
	{ "W",   "Waiting task",		 THREAD_STATE | USER_THREAD_STATE },
	{ "WA",  "Waiting all tasks",		 THREAD_STATE | USER_THREAD_STATE },
	{ "No",  "Nothing",			 THREAD_STATE | USER_THREAD_STATE },
};

static const char *get_state_name(const char *short_name, uint32_t states)
{
	unsigned i;

	for (i = 0; i < sizeof(states_list) / sizeof(states_list[0]); i++)
		if ((states_list[i].flags & states) &&
		    !strcmp(states_list[i].short_name, short_name))
			return states_list[i].long_name;
	return short_name;
}

static double get_event_time_stamp(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	return (((double)(ev->time-options->file_offset))/1000000.0);
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

static int nworkers = 0;

struct worker_entry
{
	UT_hash_handle hh;
	unsigned long tid;
	int workerid;
	int sync; /* Set only for workers which are part of the same set, i.e. on thread drivers several workers */
} *worker_ids;

static int register_thread(unsigned long nodeid, unsigned long tid, int workerid, int sync)
{
	struct worker_entry *entry = NULL;

	tid = nodeid*tid+tid;

	HASH_FIND(hh, worker_ids, &tid, sizeof(tid), entry);

	/* only register a thread once */
	if (entry)
		return 0;

	_STARPU_MALLOC(entry, sizeof(*entry));
	entry->tid = tid;
	entry->workerid = workerid;
	entry->sync = sync;

	HASH_ADD(hh, worker_ids, tid, sizeof(tid), entry);
	return 1;
}

static int register_worker_id(unsigned long nodeid, unsigned long tid, int workerid, int sync)
{
	nworkers++;
	STARPU_ASSERT_MSG(workerid < STARPU_NMAXWORKERS, "Too many workers in this trace, please increase in ./configure invocation the maximum number of CPUs and GPUs to the same value as was used for execution");

	return register_thread(nodeid, tid, workerid, sync);
}
static int prefixTOnodeid (const char *prefix)
{
	//if we are a single-node trace, prefix is empty, so return 0
	if (strcmp(prefix, "")==0) return 0;

	char *str;
	_STARPU_MALLOC(str, sizeof(char)*strlen(prefix));
	strncpy (str, prefix, strlen(prefix));
	str[strlen(prefix)-1] = '\0';
	unsigned long nodeid = atoi(str);
	free(str);
	return nodeid;
}

/* Register user threads if not done already */
static void register_user_thread(double timestamp, unsigned long tid, const char *prefix)
{
	if (register_thread(prefixTOnodeid(prefix), tid, -1, 0) && out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char program_container[STARPU_POTI_STR_LEN];
		program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
		char new_thread_container_alias[STARPU_POTI_STR_LEN];
		thread_container_alias (new_thread_container_alias, STARPU_POTI_STR_LEN, prefix, tid);
		char new_thread_container_name[STARPU_POTI_STR_LEN];
		snprintf(new_thread_container_name, sizeof(new_thread_container_name), "%sUserThread%lu", prefix, tid);
		poti_CreateContainer(timestamp, new_thread_container_alias, "UT", program_container, new_thread_container_alias);
#else
		fprintf(out_paje_file, "7	%.9f	%st%lu	UT	%sp	%sUserThread%lu\n",
			timestamp, prefix, tid, prefix, prefix, tid);
#endif
	}
}

static void register_mpi_thread(unsigned long nodeid, unsigned long tid)
{
	int ret = register_thread(nodeid, tid, -2, 0);
	STARPU_ASSERT(ret == 1);
}

static int find_worker_id(unsigned long nodeid, unsigned long tid)
{
	struct worker_entry *entry;

	tid = nodeid*tid+tid;

	HASH_FIND(hh, worker_ids, &tid, sizeof(tid), entry);
	if (!entry)
		return -1;

	return entry->workerid;
}

/* check whether this thread manages several workers */
static int find_sync(unsigned long nodeid, unsigned long tid)
{
	struct worker_entry *entry;

	tid = nodeid*tid+tid;

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

static void memnode_push_state(double time, const char *prefix, unsigned int memnodeid, const char *name)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	memmanager_container_alias(container, STARPU_POTI_STR_LEN, prefix, memnodeid);
	poti_PushState(time, container, "MS", name);
#else
	fprintf(out_paje_file, "11	%.9f	%smm%u	MS	%s\n", time, prefix, memnodeid, name);
#endif
}

static void memnode_pop_state(double time, const char *prefix, unsigned int memnodeid)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	memmanager_container_alias(container, STARPU_POTI_STR_LEN, prefix, memnodeid);
	poti_PopState(time, container, "MS");
#else
	fprintf(out_paje_file, "12	%.9f	%smm%u	MS\n", time, prefix, memnodeid);
#endif
}

static void memnode_event(double time, const char *prefix, unsigned int memnodeid, const char *name, unsigned long handle, unsigned long info, unsigned long size, unsigned int dest, struct starpu_fxt_options *options)
{
	if (!options->memory_states)
		return;
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	char p_handle[STARPU_POTI_STR_LEN];
	memmanager_container_alias(container, STARPU_POTI_STR_LEN, prefix, memnodeid);
	snprintf(p_handle, sizeof(p_handle), "%lx", handle);

#ifdef HAVE_POTI_USER_NEWEVENT
	char p_dest[STARPU_POTI_STR_LEN];
	char p_info[STARPU_POTI_STR_LEN];
	char p_size[STARPU_POTI_STR_LEN];

	memmanager_container_alias(p_dest, STARPU_POTI_STR_LEN, prefix, dest);
	snprintf(p_info, sizeof(p_info), "%lu", info);
	snprintf(p_size, sizeof(p_size), "%lu", size);

	poti_user_NewEvent(_starpu_poti_MemoryEvent, time, container, name, "0", 4,
			   p_handle, p_info, p_size, p_dest);
#else
	poti_NewEvent(time, container, name, p_handle);
#endif
#else
	fprintf(out_paje_file, "22    %.9f    %s %smm%u  0 %lx %lu %lu %smm%u\n", time, name, prefix, memnodeid, handle, info, size, prefix, dest);
#endif
}

static void worker_set_state(double time, const char *prefix, long unsigned int workerid, const char *name)
{
	if (fut_keymask == FUT_KEYMASK0)
		return;
	if (!out_paje_file)
		return;
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, workerid);
	poti_SetState(time, container, "WS", name);
#else
	fprintf(out_paje_file, "10	%.9f	%sw%lu	WS	\"%s\"\n", time, prefix, workerid, name);
#endif
}

static void worker_push_state(double time, const char *prefix, long unsigned int workerid, const char *name)
{
	if (fut_keymask == FUT_KEYMASK0)
		return;
	if (!out_paje_file)
		return;
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
	if (fut_keymask == FUT_KEYMASK0)
		return;
	if (!out_paje_file)
		return;
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
	if (find_sync(prefixTOnodeid(prefix), threadid))
		/* Unless using worker sets, collapse thread and worker */
		return worker_set_state(time, prefix, find_worker_id(prefixTOnodeid(prefix), threadid), name);
	if (!out_paje_file)
		return;

#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
	poti_SetState(time, container, "S", name);
#else
	fprintf(out_paje_file, "10	%.9f	%st%lu	S	%s\n", time, prefix, threadid, name);
#endif
}

#if 0
/* currently unused */
static void user_thread_set_state(double time, const char *prefix, long unsigned int threadid, const char *name)
{
	register_user_thread(time, threadid, prefix);
	if (!out_paje_file)
		return;
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
	poti_SetState(time, container, "US", name);
#else
	fprintf(out_paje_file, "10	%.9f	%st%lu	US	%s\n", time, prefix, threadid, name);
#endif
}
#endif

static void user_thread_push_state(double time, const char *prefix, long unsigned int threadid, const char *name)
{
	register_user_thread(time, threadid, prefix);
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];
		thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
		poti_PushState(time, container, "US", name);
#else
		fprintf(out_paje_file, "11	%.9f	%st%lu	US	%s\n", time, prefix, threadid, name);
#endif
	}
}

static void user_thread_pop_state(double time, const char *prefix, long unsigned int threadid)
{
	register_user_thread(time, threadid, prefix);
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];
		thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
		poti_PopState(time, container, "US");
#else
		fprintf(out_paje_file, "12	%.9f	%st%lu	US\n", time, prefix, threadid);
#endif
	}
}

static void thread_push_state(double time, const char *prefix, long unsigned int threadid, const char *name)
{
	if (find_sync(prefixTOnodeid(prefix), threadid))
		/* Unless using worker sets, collapse thread and worker */
		return worker_push_state(time, prefix, find_worker_id(prefixTOnodeid(prefix), threadid), name);

	if (!out_paje_file)
		return;
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
	if (find_sync(prefixTOnodeid(prefix), threadid))
		/* Unless using worker sets, collapse thread and worker */
		return worker_pop_state(time, prefix, find_worker_id(prefixTOnodeid(prefix), threadid));

	if (!out_paje_file)
		return;
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	thread_container_alias(container, STARPU_POTI_STR_LEN, prefix, threadid);
	poti_PopState(time, container, "S");
#else
	fprintf(out_paje_file, "12	%.9f	%st%lu	S\n", time, prefix, threadid);
#endif
}

static void worker_set_detailed_state(double time, const char *prefix, long unsigned int workerid, const char *name, unsigned long size, const char *parameters, unsigned long footprint, unsigned long long tag, unsigned long job_id, double gflop, unsigned X, unsigned Y, unsigned Z STARPU_ATTRIBUTE_UNUSED, long iteration, long subiteration, struct starpu_fxt_options *options)
{
	struct task_info *task = get_task(job_id, options->file_rank);
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, workerid);
	char size_str[STARPU_POTI_STR_LEN];
	char parameters_str[STARPU_POTI_STR_LEN];
	char footprint_str[STARPU_POTI_STR_LEN];
	char tag_str[STARPU_POTI_STR_LEN];
	char jobid_str[STARPU_POTI_STR_LEN];
	char submitorder_str[STARPU_POTI_STR_LEN];
	char gflop_str[STARPU_POTI_STR_LEN];
	char X_str[STARPU_POTI_STR_LEN], Y_str[STARPU_POTI_STR_LEN], Z_str[STARPU_POTI_STR_LEN];
	char iteration_str[STARPU_POTI_STR_LEN], subiteration_str[STARPU_POTI_STR_LEN];

	snprintf(size_str, sizeof(size_str), "%lu", size);
	snprintf(parameters_str, sizeof(parameters_str), "%s", parameters);
	snprintf(footprint_str, sizeof(footprint_str), "%08lx", footprint);
	snprintf(tag_str, sizeof(tag_str), "%016llx", tag);
	snprintf(jobid_str, sizeof(jobid_str), "%s%lu", prefix, job_id);
	snprintf(submitorder_str, sizeof(submitorder_str), "%s%lu", prefix, task->submit_order);
	snprintf(gflop_str, sizeof(gflop_str), "%f", gflop);
	snprintf(X_str, sizeof(X_str), "%u", X);
	snprintf(Y_str, sizeof(Y_str), "%u", Y);
	snprintf(Z_str, sizeof(Z_str), "%u", Z);
	snprintf(iteration_str, sizeof(iteration_str), "%ld", iteration);
	snprintf(subiteration_str, sizeof(subiteration_str), "%ld", subiteration);

#ifdef HAVE_POTI_INIT_CUSTOM
	poti_user_SetState(_starpu_poti_extendedSetState, time, container, "WS", name, 11, size_str,
			   parameters_str,
			   footprint_str,
			   tag_str,
			   jobid_str,
			   submitorder_str,
			   gflop_str,
			   X_str,
			   Y_str,
			   /* Z_str, */
			   iteration_str,
			   subiteration_str);
#else
	poti_SetState(time, container, "WS", name);
#endif
#else
	fprintf(out_paje_file, "20	%.9f	%sw%lu	WS	%s	%lu	%s	%08lx	%016llx	%s%lu	%s%lu	%f	%u	%u	"/*"%u	"*/"%ld	%ld\n", time, prefix, workerid, name, size, parameters, footprint, tag, prefix, job_id, prefix, task->submit_order, gflop, X, Y, /*Z,*/ iteration, subiteration);
#endif
}

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

static void mpicommthread_push_state(double time, const char *prefix, const char *name)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	mpicommthread_container_alias(container, STARPU_POTI_STR_LEN, prefix);
	poti_PushState(time, container, "CtS", name);
#else
	fprintf(out_paje_file, "11	%.9f	%smpict	CtS 	%s\n", time, prefix, name);
#endif
}

static void mpicommthread_pop_state(double time, const char *prefix)
{
#ifdef STARPU_HAVE_POTI
	char container[STARPU_POTI_STR_LEN];
	mpicommthread_container_alias(container, STARPU_POTI_STR_LEN, prefix);
	poti_PopState(time, container, "CtS");
#else
	fprintf(out_paje_file, "12	%.9f	%smpict	CtS\n", time, prefix);
#endif
}

static void recfmt_dump_state(double time, const char *event, int workerid, long int threadid, const char *name, const char *type)
{
	fprintf(trace_file, "E: %s\n", event);
	if (name)
		fprintf(trace_file, "N: %s\n", name);
	if (type)
		fprintf(trace_file, "C: %s\n", type);
	fprintf(trace_file, "W: %d\n", workerid);
	if (threadid == -1)
		fprintf(trace_file, "T: -1\n");
	else
		fprintf(trace_file, "T: %ld\n", threadid);
	fprintf(trace_file, "S: %f\n", time);
	fprintf(trace_file, "\n");
}

static void recfmt_set_state(double time, int workerid, long int threadid, const char *name, const char *type)
{
	recfmt_dump_state(time, "SetState", workerid, threadid, name, type);
}

static void recfmt_push_state(double time, int workerid, long unsigned int threadid, const char *name, const char *type)
{
	recfmt_dump_state(time, "PushState", workerid, threadid, name, type);
}

static void recfmt_pop_state(double time, int workerid, long unsigned int threadid)
{
	recfmt_dump_state(time, "PopState", workerid, threadid, NULL, NULL);
}

static void recfmt_worker_set_state(double time, int workerid, const char *name, const char *type)
{
	const char *state_name;

	/* Special case for task events. */
	if (!strcmp(type, "Task"))
		state_name = name;
	else
		state_name = get_state_name(name, WORKER_STATE);
	recfmt_set_state(time, workerid, -1, state_name, type);
}

static void recfmt_thread_set_state(double time, unsigned long nodeid, long unsigned int threadid, const char *name, const char *type)
{
	const char *state_name;

	/* Special case for the end event which is somehow a fake. */
	if (!strcmp(name, "End") && !type)
		state_name = name;
	else
		state_name = get_state_name(name, THREAD_STATE);

	recfmt_set_state(time, find_worker_id(nodeid, threadid), threadid, state_name, type);
}

static void recfmt_thread_push_state(double time, unsigned long nodeid, long unsigned int threadid, const char *name, const char *type)
{
	const char *state_name = get_state_name(name, THREAD_STATE);
	recfmt_push_state(time, find_worker_id(nodeid, threadid), threadid, state_name, type);
}

static void recfmt_thread_pop_state(double time, unsigned long nodeid, long unsigned int threadid)
{
	recfmt_pop_state(time, find_worker_id(nodeid, threadid), threadid);
}

static void recfmt_mpicommthread_set_state(double time, const char *name)
{
	const char *state_name = get_state_name(name, COMM_THREAD_STATE);
	recfmt_set_state(time, -1, 0, state_name, "MPI"); /* XXX */
}

static void recfmt_mpicommthread_push_state(double time, const char *name)
{
	const char *state_name = get_state_name(name, COMM_THREAD_STATE);
	recfmt_push_state(time, -1, 0, state_name, "MPI"); /* XXX */
}

static void recfmt_mpicommthread_pop_state(double time)
{
	recfmt_pop_state(time, -1, 0);
}

static void recfmt_user_thread_push_state(double time, long unsigned threadid, const char *name, const char *type)
{
	const char *state_name = get_state_name(name, USER_THREAD_STATE);
	recfmt_push_state(time, -1, threadid, state_name, type);
}

static void recfmt_user_thread_pop_state(double time, long unsigned threadid)
{
	recfmt_pop_state(time, -1, threadid);
}

/*
 *	Initialization
 */

static void handle_new_mem_node(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;

	if (out_paje_file)
	{
		double now = get_event_time_stamp(ev, options);
#ifdef STARPU_HAVE_POTI
		char program_container[STARPU_POTI_STR_LEN];
		program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
		char new_memnode_container_alias[STARPU_POTI_STR_LEN], new_memnode_container_name[STARPU_POTI_STR_LEN];
		char new_memmanager_container_alias[STARPU_POTI_STR_LEN], new_memmanager_container_name[STARPU_POTI_STR_LEN];
		memnode_container_alias (new_memnode_container_alias, STARPU_POTI_STR_LEN, prefix, ev->param[0]);
		/* TODO: ramkind */
		snprintf(new_memnode_container_name, sizeof(new_memnode_container_name), "%sMEMNODE%"PRIu64"", prefix, ev->param[0]);
		poti_CreateContainer(now, new_memnode_container_alias, "Mn", program_container, new_memnode_container_name);

		memmanager_container_alias (new_memmanager_container_alias, STARPU_POTI_STR_LEN, prefix, ev->param[0]);
		/* TODO: ramkind */
		snprintf(new_memmanager_container_name, sizeof(new_memmanager_container_name), "%sMEMMANAGER%"PRIu64"", prefix, ev->param[0]);
		poti_CreateContainer(now, new_memmanager_container_alias, "Mm", new_memnode_container_alias, new_memmanager_container_name);
#else
		fprintf(out_paje_file, "7	%.9f	%smn%"PRIu64"	Mn	%sp	%sMEMNODE%"PRIu64"\n", now, prefix, ev->param[0], prefix, options->file_prefix, ev->param[0]);
		fprintf(out_paje_file, "7	%.9f	%smm%"PRIu64"	Mm	%smn%"PRIu64"	%sMEMMANAGER%"PRIu64"\n", now, prefix, ev->param[0], prefix, ev->param[0], options->file_prefix, ev->param[0]);
#endif

		if (!options->no_bus)
		{
#ifdef STARPU_HAVE_POTI
			poti_SetVariable(now, new_memmanager_container_alias, "use", 0.0);
			poti_SetVariable(now, new_memmanager_container_alias, "bwi_mm", 0.0);
			poti_SetVariable(now, new_memmanager_container_alias, "bwo_mm", 0.0);
#else
			fprintf(out_paje_file, "13	%.9f	%smm%"PRIu64"	use	0.0\n", now, prefix, ev->param[0]);
			fprintf(out_paje_file, "13	%.9f	%smm%"PRIu64"	bwi_mm	0.0\n", now, prefix, ev->param[0]);
			fprintf(out_paje_file, "13	%.9f	%smm%"PRIu64"	bwo_mm	0.0\n", now, prefix, ev->param[0]);
#endif
		}
	}
}

/*
 * Function that creates a synthetic stream id based on the order they appear from the trace
 */
static int create_ordered_stream_id (int nodeid, int devid)
{
	static int stable[MAX_MPI_NODES][STARPU_MAXCUDADEVS];
	STARPU_ASSERT(nodeid < MAX_MPI_NODES);
	STARPU_ASSERT(devid < STARPU_MAXCUDADEVS);
	return stable[nodeid][devid]++;
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
	long unsigned int threadid = ev->param[6];
	int new_thread;

	new_thread = register_worker_id(prefixTOnodeid(prefix), threadid, workerid, set);

	char *kindstr = "";
	struct starpu_perfmodel_arch arch;
	arch.ndevices = 1;
	_STARPU_MALLOC(arch.devices, sizeof(struct starpu_perfmodel_device));

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
		case _STARPU_FUT_MPI_KEY:
			set_next_mpi_ms_worker_color(workerid);
			kindstr = "mpi_ms";
			arch.devices[0].type = STARPU_MPI_MS_WORKER;
			arch.devices[0].devid = devid;
			arch.devices[0].ncores = 1;
			break;

		default:
			STARPU_ABORT();
	}
	double now = get_event_time_stamp(ev, options);

	if (out_paje_file)
	{
		char new_worker_container_name[STARPU_TRACE_STR_LEN];
		if (arch.devices[0].type == STARPU_CUDA_WORKER)
		{
			// If CUDA, workers might be streams, so create an unique name for each of them
			int streamid = create_ordered_stream_id (prefixTOnodeid(prefix), devid);
			snprintf(new_worker_container_name, sizeof(new_worker_container_name), "%s%s%d_%d", prefix, kindstr, devid, streamid);
		}
		else
		{
			// If not CUDA, we suppose worker name is the prefix, the kindstr, and the devid
			snprintf(new_worker_container_name, sizeof(new_worker_container_name), "%s%s%d", prefix, kindstr, devid);
		}
#ifdef STARPU_HAVE_POTI
		char new_thread_container_alias[STARPU_POTI_STR_LEN];
		thread_container_alias (new_thread_container_alias, STARPU_POTI_STR_LEN, prefix, threadid);
		char new_worker_container_alias[STARPU_POTI_STR_LEN];
		worker_container_alias (new_worker_container_alias, STARPU_POTI_STR_LEN, prefix, workerid);
		char memnode_container[STARPU_POTI_STR_LEN];
		memnode_container_alias(memnode_container, STARPU_POTI_STR_LEN, prefix, nodeid);
		char new_thread_container_name[STARPU_POTI_STR_LEN];
		snprintf(new_thread_container_name, sizeof(new_thread_container_name), "%sT%d", prefix, bindid);
		if (new_thread)
			poti_CreateContainer(now, new_thread_container_alias, "T", memnode_container, new_thread_container_name);
		poti_CreateContainer(now, new_worker_container_alias, "W", new_thread_container_alias, new_worker_container_name);
		if (!options->no_flops)
			poti_SetVariable(now, new_worker_container_alias, "gf", 0.0);
#else
		if (new_thread)
			fprintf(out_paje_file, "7	%.9f	%st%lu	T	%smn%d	%sT%d\n",
				now, prefix, threadid, prefix, nodeid, prefix, bindid);
		fprintf(out_paje_file, "7	%.9f	%sw%d	W	%st%lu	%s\n",
			now, prefix, workerid, prefix, threadid, new_worker_container_name);
		if (!options->no_flops)
			fprintf(out_paje_file, "13	%.9f	%sw%d	gf	0.0\n",
				now, prefix, workerid);
#endif
	}

	/* start initialization */
	thread_set_state(now, prefix, threadid, "In");
	if (trace_file)
		recfmt_thread_set_state(now, prefixTOnodeid(prefix), threadid, "In", "Runtime");

	if (activity_file)
		fprintf(activity_file, "name\t%d\t%s %d\n", workerid, kindstr, devid);

	snprintf(options->worker_names[workerid], sizeof(options->worker_names[workerid])-1, "%s %d", kindstr, devid);
	options->worker_names[workerid][sizeof(options->worker_names[workerid])-1] = 0;
	options->worker_archtypes[workerid] = arch;
}

static void handle_worker_init_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	int worker;

	if (ev->nb_params < 2)
	{
		worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
		STARPU_ASSERT(worker >= 0);
	}
	else
		worker = ev->param[1];

	thread_set_state(get_event_time_stamp(ev, options), prefix, ev->param[0], "B");
	if (trace_file)
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0], "B", "Runtime");

	worker_set_state(get_event_time_stamp(ev, options), prefix, worker, "I");
	if (trace_file)
		recfmt_worker_set_state(get_event_time_stamp(ev, options), worker, "I", "Other");

	/* Initilize the accumulated time counters */
	last_activity_flush_timestamp[worker] = get_event_time_stamp(ev, options);
	accumulated_sleep_time[worker] = 0.0;
	accumulated_exec_time[worker] = 0.0;
}

static void handle_worker_deinit_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	long unsigned int threadid = ev->param[0];

	thread_set_state(get_event_time_stamp(ev, options), prefix, threadid, "D");
	if (trace_file)
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), threadid, "D", "Runtime");
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
	if (trace_file)
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[1], "End", NULL);
}

#ifdef STARPU_HAVE_POTI
static void create_paje_state_color(char *name, char *type, int ctx, float red, float green, float blue)
{
	char color[STARPU_POTI_STR_LEN];
	char alias[STARPU_POTI_STR_LEN];
	snprintf(color, sizeof(color), "%f %f %f", red, green, blue);
	if (ctx)
	{
		snprintf(alias, sizeof(alias), "%s_%d", name, ctx);
	}
	else
	{
		snprintf(alias, sizeof(alias), "%s", name);
	}
	poti_DefineEntityValue(alias, type, name, color);
}
#endif

static void create_paje_state_if_not_found(char *name, unsigned color, struct starpu_fxt_options *options)
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
	entry->name = strdup(name);
	STARPU_ASSERT(entry->name);

	_starpu_symbol_name_list_push_front(&symbol_list, entry);

	/* choose some color ... that's disguting yes */
	unsigned hash_symbol_red = get_color_symbol_red(name);
	unsigned hash_symbol_green = get_color_symbol_green(name);
	unsigned hash_symbol_blue = get_color_symbol_blue(name);

	uint32_t hash_sum = hash_symbol_red + hash_symbol_green + hash_symbol_blue;

	float red, green, blue;
	if (color != 0)
	{
		red = color / 0x100 / 0x100;
		green = (color / 0x100) & 0xff;
		blue = color & 0xff;
	}
	else if (options->per_task_colour)
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
		create_paje_state_color(name, "WS", 0, red, green, blue);
		int i;
		for(i = 1; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			char ctx[10];
			snprintf(ctx, sizeof(ctx), "Ctx%d", i);
			if(i%10 == 1)
				create_paje_state_color(name, ctx, i, 1.0, 0.39, 1.0);
			if(i%10 == 2)
				create_paje_state_color(name, ctx, i, .0, 1.0, 0.0);
			if(i%10 == 3)
				create_paje_state_color(name, ctx, i, 1.0, 1.0, .0);
			if(i%10 == 4)
				create_paje_state_color(name, ctx, i, .0, 0.95, 1.0);
			if(i%10 == 5)
				create_paje_state_color(name, ctx, i, .0, .0, .0);
			if(i%10 == 6)
				create_paje_state_color(name, ctx, i, .0, .0, 0.5);
			if(i%10 == 7)
				create_paje_state_color(name, ctx, i, 0.41, 0.41, 0.41);
			if(i%10 == 8)
				create_paje_state_color(name, ctx, i, 1.0, .0, 1.0);
			if(i%10 == 9)
				create_paje_state_color(name, ctx, i, .0, .0, 1.0);
			if(i%10 == 0)
				create_paje_state_color(name, ctx, i, 0.6, 0.80, 50.0);

		}
/* 		create_paje_state_color(name, "Ctx1", 1.0, 0.39, 1.0); */
/* 		create_paje_state_color(name, "Ctx2", .0, 1.0, 0.0); */
/* 		create_paje_state_color(name, "Ctx3", 1.0, 1.0, .0); */
/* 		create_paje_state_color(name, "Ctx4", .0, 0.95, 1.0); */
/* 		create_paje_state_color(name, "Ctx5", .0, .0, .0); */
/* 		create_paje_state_color(name, "Ctx6", .0, .0, 0.5); */
/* 		create_paje_state_color(name, "Ctx7", 0.41, 0.41, 0.41); */
/* 		create_paje_state_color(name, "Ctx8", 1.0, .0, 1.0); */
/* 		create_paje_state_color(name, "Ctx9", .0, .0, 1.0); */
/* 		create_paje_state_color(name, "Ctx10", 0.6, 0.80, 0.19); */
#else
		fprintf(out_paje_file, "6	%s	WS	%s	\"%f %f %f\" \n", name, name, red, green, blue);
		int i;
		for(i = 1; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			if(i%10 == 1)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\"1.0 0.39 1.0\" \n", name, i, i, name);
			if(i%10 == 2)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\".0 1.0 .0\" \n", name, i, i, name);
			if(i%10 == 3)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\"0.87 0.87 .0\" \n", name, i, i, name);
			if(i%10 == 4)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\".0 0.95 1.0\" \n", name, i, i, name);
			if(i%10 == 5)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\".0 .0 .0\" \n", name, i, i, name);
			if(i%10 == 6)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\".0 .0 0.5\" \n", name, i, i, name);
			if(i%10 == 7)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\"0.41 0.41 0.41\" \n", name, i, i, name);
			if(i%10 == 8)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\"1.0 .0 1.0\" \n", name, i, i, name);
			if(i%10 == 9)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\".0 .0 1.0\" \n", name, i, i, name);
			if(i%10 == 0)
				fprintf(out_paje_file, "6	%s_%d	Ctx%d	%s	\"0.6 0.80 0.19\" \n", name, i, i, name);

		}

/* 		fprintf(out_paje_file, "6	%s	Ctx1	%s	\"1.0 0.39 1.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx2	%s	\".0 1.0 .0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx3	%s	\"0.87 0.87 .0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx4	%s	\".0 0.95 1.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx5	%s	\".0 .0 .0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx6	%s	\".0 .0 0.5\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx7	%s	\"0.41 0.41 0.41\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx8	%s	\"1.0 .0 1.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx9	%s	\".0 .0 1.0\" \n", name, name); */
/* 		fprintf(out_paje_file, "6	%s	Ctx10	%s	\"0.6 0.80 0.19\" \n", name, name); */
#endif
	}

}


static void handle_start_codelet_body(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker = ev->param[2];
	int node = ev->param[3];

	if (worker < 0) return;

	unsigned long has_name = ev->param[4];
	char *name = has_name?get_fxt_string(ev, 5):"unknown";

	snprintf(_starpu_last_codelet_symbol[worker], sizeof(_starpu_last_codelet_symbol[worker])-1, "%s", name);
	_starpu_last_codelet_symbol[worker][sizeof(_starpu_last_codelet_symbol[worker])-1] = 0;
	last_codelet_parameter[worker] = 0;

	double start_codelet_time = get_event_time_stamp(ev, options);
	double last_start_codelet_time = last_codelet_start[worker];
	last_codelet_start[worker] = start_codelet_time;

	struct task_info *task = get_task(ev->param[0], options->file_rank);
	create_paje_state_if_not_found(name, task->color, options);

	task->start_time = start_codelet_time;
	task->workerid = worker;
	task->name = strdup(name);
	task->node = node;

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
			snprintf(ctx, sizeof(ctx), "Ctx%u", sched_ctx);
			worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, ev->param[2]);
			poti_SetState(start_codelet_time, container, ctx, name);
#else
			fprintf(out_paje_file, "10	%.9f	%sw%"PRIu64"	Ctx%d	\"%s\"\n", start_codelet_time, prefix, ev->param[2], sched_ctx, name);
#endif
		}
	}
	if (trace_file)
		recfmt_worker_set_state(start_codelet_time, ev->param[2], name, "Task");

	struct _starpu_computation *comp = ongoing_computation[worker];
	if (!comp)
	{
		/* First task for this worker */
		comp = ongoing_computation[worker] = _starpu_computation_new();
		comp->peer = NULL;
		comp->comp_start = start_codelet_time;
		if (!options->no_flops)
			_starpu_computation_list_push_back(&computation_list, comp);
	}
	else if (options->no_smooth ||
			(start_codelet_time - last_codelet_end[worker]) >=
			IDLE_FACTOR * (last_codelet_end[worker] - last_start_codelet_time))
	{
		/* Long idle period, move previously-allocated comp to now */
		comp->comp_start = start_codelet_time;
		if (!options->no_flops)
		{
			_starpu_computation_list_erase(&computation_list, comp);
			_starpu_computation_list_push_back(&computation_list, comp);
		}
	}
}

static void handle_model_name(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	struct task_info *task = get_task(ev->param[0], options->file_rank);
	char *name = get_fxt_string(ev, 1);
	task->model_name = strdup(name);
}

static void handle_codelet_data(struct fxt_ev_64 *ev STARPU_ATTRIBUTE_UNUSED, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	int worker = ev->param[0];
	if (worker < 0) return;
	int num = last_codelet_parameter[worker]++;
	if (num >= MAX_PARAMETERS)
		return;
	char *name = get_fxt_string(ev, 1);
	snprintf(last_codelet_parameter_description[worker][num], sizeof(last_codelet_parameter_description[worker][num])-1, "%s", name);
	last_codelet_parameter_description[worker][num][sizeof(last_codelet_parameter_description[worker][num])-1] = 0;
}

static void handle_codelet_data_handle(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	struct task_info *task = get_task(ev->param[0], options->file_rank);
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
	{
		_STARPU_REALLOC(task->data, sizeof(*task->data) * alloc);
	}
	task->data[task->ndata].handle = ev->param[1];
	task->data[task->ndata].size = ev->param[2];
	task->data[task->ndata].mode = ev->param[3];
	task->ndata++;
}

static void handle_codelet_details(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker = ev->param[5];
	unsigned long job_id = ev->param[6];

	if (worker < 0) return;

	char parameters[256];
	size_t eaten = 0;
	if (!last_codelet_parameter[worker])
		eaten += snprintf(parameters + eaten, sizeof(parameters) - eaten - 1, "nodata");
	else
	{
		int i;
		for (i = 0; i < last_codelet_parameter[worker] && i < MAX_PARAMETERS; i++)
		{
			eaten += snprintf(parameters + eaten, sizeof(parameters) - eaten - 1, "%s%s", i?" ":"", last_codelet_parameter_description[worker][i]);
		}
	}
	parameters[sizeof(parameters)-1] = 0;

	struct task_info *task = get_task(job_id, options->file_rank);
	task->parameters = strdup(parameters);
	task->footprint = ev->param[2];
	task->kflops = ev->param[3];
	task->tag = ev->param[4];

	unsigned i, X = 0, Y = 0, Z = 0;
	for (i = 0; i < task->ndata; i++)
	{
		if (task->data[i].mode & STARPU_W)
		{
			struct data_info *data = get_data(task->data[i].handle, options->file_rank);
			if (data->dimensions >= 1)
				X = data->dims[0];
			if (data->dimensions >= 2)
				Y = data->dims[1];
			if (data->dimensions >= 3)
				Z = data->dims[2];
			break;
		}
	}

	if (out_paje_file)
	{
		char *prefix = options->file_prefix;
		unsigned sched_ctx = ev->param[0];

		/* Paje won't like spaces or tabs, replace with underscores */
		char *c;
		for (c = parameters; *c; c++)
			if ((*c == ' ') || (*c == '\t'))
				*c = '_';

		worker_set_detailed_state(last_codelet_start[worker], prefix, worker, _starpu_last_codelet_symbol[worker], ev->param[1], parameters, ev->param[2], ev->param[4], job_id, ((double) task->kflops) / 1000000, X, Y, Z, task->iterations[0], task->iterations[1], options);
		if (sched_ctx != 0)
		{
#ifdef STARPU_HAVE_POTI
			char container[STARPU_POTI_STR_LEN];
			char typectx[STARPU_POTI_STR_LEN];
			snprintf(typectx, sizeof(typectx), "Ctx%u", sched_ctx);
			worker_container_alias(container, sizeof(container), prefix, worker);
			poti_SetState(last_codelet_start[worker], container, typectx, _starpu_last_codelet_symbol[worker]);

			char name[STARPU_POTI_STR_LEN];
			snprintf(name, sizeof(name), "%s", _starpu_last_codelet_symbol[worker]);

			char size_str[STARPU_POTI_STR_LEN];
			char parameters_str[STARPU_POTI_STR_LEN];
			char footprint_str[STARPU_POTI_STR_LEN];
			char tag_str[STARPU_POTI_STR_LEN];
			char jobid_str[STARPU_POTI_STR_LEN];
			char submitorder_str[STARPU_POTI_STR_LEN];
			snprintf(size_str, sizeof(size_str), "%ld", ev->param[1]);
			snprintf(parameters_str, sizeof(parameters_str), "%s", parameters);
			snprintf(footprint_str, sizeof(footprint_str), "%08lx", ev->param[2]);
			snprintf(tag_str, sizeof(tag_str), "%016lx", ev->param[4]);
			snprintf(jobid_str, sizeof(jobid_str), "%s%lu", prefix, job_id);
			snprintf(submitorder_str, sizeof(submitorder_str), "%s%lu", prefix, task->submit_order);

#ifdef HAVE_POTI_INIT_CUSTOM
			poti_user_SetState(_starpu_poti_semiExtendedSetState, last_codelet_start[worker], container, typectx, name, 6, size_str,
					   parameters_str,
					   footprint_str,
					   tag_str,
					   jobid_str,
					   submitorder_str);
#else
			poti_SetState(last_codelet_start[worker], container, typectx, name);
#endif
#else
			fprintf(out_paje_file, "21	%.9f	%sw%d	Ctx%u	\"%s\"	%ld	%s	%08lx	%016lx	%s%lu	%s%lu\n", last_codelet_start[worker], prefix, worker, sched_ctx, _starpu_last_codelet_symbol[worker], ev->param[1], parameters,  ev->param[2], ev->param[4], prefix, job_id, prefix, task->submit_order);
#endif
		}
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
	double last_end_codelet_time = last_codelet_end[worker];
	last_codelet_end[worker] = end_codelet_time;

	size_t codelet_size = ev->param[1];
	uint32_t codelet_hash = ev->param[2];
	long unsigned int threadid = ev->param[4];
	char *name = get_fxt_string(ev, 4);

	const char *state = "I";
	if (find_sync(prefixTOnodeid(prefix), threadid))
		state = "B";

	worker_set_state(end_codelet_time, prefix, worker, state);
	if (trace_file)
		recfmt_worker_set_state(end_codelet_time, worker, state, "Other");

	struct task_info *task = get_task(ev->param[0], options->file_rank);

	get_task(ev->param[0], options->file_rank)->end_time = end_codelet_time;
	update_accumulated_time(worker, 0.0, end_codelet_time - task->start_time, end_codelet_time, 0);

	struct _starpu_computation *peer = ongoing_computation[worker];
	double gflops_start = peer->comp_start;
	double codelet_length;
	double gflops;
	struct _starpu_computation *comp;

	codelet_length = end_codelet_time - gflops_start;
	gflops = (((double)task->kflops) / 1000000) / (codelet_length / 1000);

	if (options->no_flops)
	{
		_starpu_computation_delete(peer);
	}
	else
	{
		if (out_paje_file)
		{
#ifdef STARPU_HAVE_POTI
			char container[STARPU_POTI_STR_LEN];
			worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, worker);
			if (gflops_start != last_end_codelet_time)
			{
				if (last_end_codelet_time != 0)
				{
					poti_SetVariable(last_end_codelet_time, container, "gf", 0.);
				}
			}
			poti_SetVariable(gflops_start, container, "gf", gflops);
#else
			if (gflops_start != last_end_codelet_time)
			{
				if (last_end_codelet_time != 0)
				{
					fprintf(out_paje_file, "13	%.9f	%sw%d	gf	%f\n",
						last_end_codelet_time, prefix, worker, 0.);
				}
			}
			fprintf(out_paje_file, "13	%.9f	%sw%d	gf	%f\n",
					gflops_start, prefix, worker, gflops);
#endif
		}

		comp = _starpu_computation_new();
		comp->comp_start = end_codelet_time;
		comp->gflops = -gflops;
		peer->gflops = +gflops;
		comp->peer = peer;
		peer->peer = comp;
		_starpu_computation_list_push_back(&computation_list, comp);
	}

	/* Prepare comp for next codelet */
	comp = _starpu_computation_new();
	comp->comp_start = end_codelet_time;
	comp->peer = NULL;
	if (!options->no_flops)
		_starpu_computation_list_push_back(&computation_list, comp);
	ongoing_computation[worker] = comp;

	if (distrib_time)
	     fprintf(distrib_time, "%s\t%s%d\t%ld\t%"PRIx32"\t%.9f\n", _starpu_last_codelet_symbol[worker],
		     prefix, worker, (unsigned long) codelet_size, codelet_hash, codelet_length);

	if (options->dumped_codelets)
	{
		dumped_codelets_count++;
		_STARPU_REALLOC(dumped_codelets, dumped_codelets_count*sizeof(struct starpu_fxt_codelet_event));

		snprintf(dumped_codelets[dumped_codelets_count - 1].symbol, sizeof(dumped_codelets[dumped_codelets_count - 1].symbol)-1, "%s", _starpu_last_codelet_symbol[worker]);
		dumped_codelets[dumped_codelets_count - 1].symbol[sizeof(dumped_codelets[dumped_codelets_count - 1].symbol)-1] = 0;
		dumped_codelets[dumped_codelets_count - 1].workerid = worker;
		snprintf(dumped_codelets[dumped_codelets_count - 1].perfmodel_archname, sizeof(dumped_codelets[dumped_codelets_count - 1].perfmodel_archname)-1, "%s", name);
		dumped_codelets[dumped_codelets_count - 1].perfmodel_archname[sizeof(dumped_codelets[dumped_codelets_count - 1].perfmodel_archname)-1] = 0;
		dumped_codelets[dumped_codelets_count - 1].size = codelet_size;
		dumped_codelets[dumped_codelets_count - 1].hash = codelet_hash;
		dumped_codelets[dumped_codelets_count - 1].time = codelet_length;
	}
	_starpu_last_codelet_symbol[worker][0] = 0;
}

static void handle_start_executing(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	long unsigned int threadid = ev->param[0];

	if (out_paje_file && !find_sync(prefixTOnodeid(prefix), threadid))
		thread_set_state(get_event_time_stamp(ev, options), prefix, threadid, "E");
	if (trace_file && !find_sync(prefixTOnodeid(prefix), threadid))
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), threadid, "E", "Runtime");
}

static void handle_end_executing(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	long unsigned int threadid = ev->param[0];

	if (out_paje_file && !find_sync(prefixTOnodeid(prefix), threadid))
		thread_set_state(get_event_time_stamp(ev, options), prefix, threadid, "B");
	if (trace_file && !find_sync(prefixTOnodeid(prefix), threadid))
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), threadid, "B", "Runtime");
}

static void handle_user_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	unsigned long code = ev->param[0];
#ifdef STARPU_HAVE_POTI
	char paje_value[STARPU_POTI_STR_LEN], container[STARPU_POTI_STR_LEN];
	snprintf(paje_value, sizeof(paje_value), "%lu", code);
#endif

	char *prefix = options->file_prefix;
	double now = get_event_time_stamp(ev, options);

	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[1]);
	if (worker < 0)
	{
		if (out_paje_file)
#ifdef STARPU_HAVE_POTI
			program_container_alias (container, STARPU_POTI_STR_LEN, prefix);
#else
			fprintf(out_paje_file, "9	%.9f	user_user_event	%sp	%lu\n", now, prefix, code);
#endif
	}
	else
	{
		if (out_paje_file)
#ifdef STARPU_HAVE_POTI
			thread_container_alias (container, STARPU_POTI_STR_LEN, prefix, ev->param[1]);
#else
			fprintf(out_paje_file, "9	%.9f	user_event	%st%"PRIu64"	%lu\n", now, prefix, ev->param[1], code);
#endif
	}
#ifdef STARPU_HAVE_POTI
	if (out_paje_file)
		poti_NewEvent(now, container, "user_event", paje_value);
#endif
}

static void handle_start_callback(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[1]);
	if (worker >= 0)
	{
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], "C");
		if (trace_file)
			recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[1], "C", "Runtime");
	}
	else if (worker == -2)
	{
		/* MPI thread */
		mpicommthread_push_state(get_event_time_stamp(ev, options), options->file_prefix, "C");
		recfmt_mpicommthread_push_state(get_event_time_stamp(ev, options), "C");
	}
	else
	{
		user_thread_push_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], "C");
		recfmt_user_thread_push_state(get_event_time_stamp(ev, options), ev->param[1], "C", "UNK"); /* XXX */
	}
}

static void handle_end_callback(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[1]);
	if (worker >= 0)
	{
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], "B");
		if (trace_file)
			recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[1], "B", "Runtime");
	}
	else if (worker == -2)
	{
		/* MPI thread */
		mpicommthread_pop_state(get_event_time_stamp(ev, options), options->file_prefix);
		recfmt_mpicommthread_pop_state(get_event_time_stamp(ev, options));
	}
	else
	{
		user_thread_pop_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1]);
		recfmt_user_thread_pop_state(get_event_time_stamp(ev, options), ev->param[1]);
	}
}

static void handle_hypervisor_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker >= 0)
	{
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "H");
		if (trace_file)
			recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0], "H", "Runtime");
	}
	else if (worker == -2)
	{
		/* MPI thread */
		mpicommthread_push_state(get_event_time_stamp(ev, options), options->file_prefix, "H");
		recfmt_mpicommthread_push_state(get_event_time_stamp(ev, options), "H");
	}
	else
	{
		user_thread_push_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], "H");
		recfmt_user_thread_push_state(get_event_time_stamp(ev, options), ev->param[1], "H", "UNK"); /* XXX */
	}
}

static void handle_hypervisor_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker >= 0)
	{
		thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "B");
		if (trace_file)
			recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0], "B", "Runtime");
	}
	else if (worker == -2)
	{
		/* MPI thread */
		mpicommthread_pop_state(get_event_time_stamp(ev, options), options->file_prefix);
		recfmt_mpicommthread_pop_state(get_event_time_stamp(ev, options));
	}
	else
	{
		user_thread_pop_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1]);
		recfmt_user_thread_pop_state(get_event_time_stamp(ev, options), ev->param[1]);
	}
}

static void handle_worker_status_on_tid(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *newstatus)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[1]);
	if (worker < 0)
		return;

	thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], newstatus);
	if (trace_file)
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[1], newstatus, "Runtime");
}

static void handle_worker_status(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *newstatus)
{
	int worker;
	worker = ev->param[1];
	if (worker < 0)
		return;

	worker_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[1], newstatus);
	if (trace_file)
		recfmt_worker_set_state(get_event_time_stamp(ev, options), ev->param[1], newstatus, "Runtime");
}

static double last_sleep_start[STARPU_NMAXWORKERS];

static void handle_worker_scheduling_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker < 0) return;

	thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "Sc");
	if (trace_file)
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0], "Sc", "Runtime");
}

static void handle_worker_scheduling_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker < 0) return;

	thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "B");
	if (trace_file)
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0], "B", "Runtime");
}

static void handle_worker_scheduling_push(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker < 0) return;

	thread_push_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "Sc");
	if (trace_file)
		recfmt_thread_push_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0], "Sc", "Runtime");
}

static void handle_worker_scheduling_pop(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker < 0) return;

	thread_pop_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0]);
	if (trace_file)
		recfmt_thread_pop_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0]);
}

static void handle_worker_sleep_start(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker < 0) return;

	double start_sleep_time = get_event_time_stamp(ev, options);
	last_sleep_start[worker] = start_sleep_time;

	thread_set_state(get_event_time_stamp(ev, options), options->file_prefix, ev->param[0], "Sl");
	if (trace_file)
		recfmt_thread_set_state(get_event_time_stamp(ev, options), prefixTOnodeid(prefix), ev->param[0], "Sl", "Other");
}

static void handle_worker_sleep_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int worker;
	char *prefix = options->file_prefix;
	worker = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	if (worker < 0) return;

	double end_sleep_timestamp = get_event_time_stamp(ev, options);

	thread_set_state(end_sleep_timestamp, options->file_prefix, ev->param[0], "B");
	if (trace_file)
		recfmt_thread_set_state(end_sleep_timestamp, prefixTOnodeid(prefix), ev->param[0], "B", "Runtime");

	double sleep_length = end_sleep_timestamp - last_sleep_start[worker];

	update_accumulated_time(worker, sleep_length, 0.0, end_sleep_timestamp, 0);
}

static void handle_data_register(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	char *prefix = options->file_prefix;
	struct data_info *data = get_data(handle, options->file_rank);
	char *description = get_fxt_string(ev, 3);

	data->size = ev->param[1];
	data->home_node = ev->param[2];
	if (description[0])
		data->description = strdup(description);

	if (out_paje_file && !options->no_events)
	{
#ifdef STARPU_HAVE_POTI
		char paje_value[STARPU_POTI_STR_LEN], container[STARPU_POTI_STR_LEN];
		snprintf(paje_value, sizeof(paje_value), "%lx", handle);
		program_container_alias (container, STARPU_POTI_STR_LEN, prefix);
		poti_NewEvent(get_event_time_stamp(ev, options), container, "register", paje_value);
#else
		fprintf(out_paje_file, "9	%.9f	register	%sp	%lx\n", get_event_time_stamp(ev, options), prefix, handle);
#endif
	}
}

static void handle_data_unregister(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	char *prefix = options->file_prefix;
	struct data_info *data = get_data(handle, options->file_rank);

	if (out_paje_file && !options->no_events)
	{
#ifdef STARPU_HAVE_POTI
		char paje_value[STARPU_POTI_STR_LEN], container[STARPU_POTI_STR_LEN];
		snprintf(paje_value, sizeof(paje_value), "%lx", handle);
		program_container_alias (container, STARPU_POTI_STR_LEN, prefix);
		poti_NewEvent(get_event_time_stamp(ev, options), container, "unregister", paje_value);
#else
		fprintf(out_paje_file, "9	%.9f	unregister	%sp	%lx\n", get_event_time_stamp(ev, options), prefix, handle);
#endif
	}

	data_dump(data);
}

static void handle_data_state(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *state)
{
	unsigned long handle = ev->param[0];
	unsigned node = ev->param[1];
	char *prefix = options->file_prefix;

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char paje_value[STARPU_POTI_STR_LEN], memnode_container[STARPU_POTI_STR_LEN];
		memmanager_container_alias(memnode_container, STARPU_POTI_STR_LEN, prefix, node);
		snprintf(paje_value, sizeof(paje_value), "%lx", handle);
		poti_NewEvent(get_event_time_stamp(ev, options), memnode_container, state, paje_value);
#else
		fprintf(out_paje_file, "9	%.9f	%s	%smm%u	%lx\n", get_event_time_stamp(ev, options), state, prefix, node, handle);
#endif
	}
}

static void handle_data_copy(void)
{
}

static void handle_data_name(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	char *name = get_fxt_string(ev, 1);
	struct data_info *data = get_data(handle, options->file_rank);

	data->name = strdup(name);
}

static void handle_data_coordinates(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	unsigned dimensions = ev->param[1];
	struct data_info *data = get_data(handle, options->file_rank);
	unsigned i;

	data->dimensions = dimensions;
	_STARPU_MALLOC(data->dims, dimensions * sizeof(*data->dims));
	for (i = 0; i < dimensions; i++)
		data->dims[i] = ev->param[i+2];
}

static void handle_data_wont_use(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	unsigned long submit_order = ev->param[1];
	unsigned long job_id = ev->param[2];

	fprintf(tasks_file, "Control: WontUse\n");
	fprintf(tasks_file, "JobId: %lu\n", job_id);
	fprintf(tasks_file, "SubmitOrder: %lu\n", submit_order);
	fprintf(tasks_file, "SubmitTime: %f\n", get_event_time_stamp(ev, options));
	fprintf(tasks_file, "Handles: %lx\n", handle);
	fprintf(tasks_file, "MPIRank: %d\n", options->file_rank);
	fprintf(tasks_file, "\n");
}

static void handle_data_doing_wont_use(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	char *prefix = options->file_prefix;
	unsigned node = STARPU_MAIN_RAM;
	const char *event = "WU";

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char paje_value[STARPU_POTI_STR_LEN], memnode_container[STARPU_POTI_STR_LEN];
		memmanager_container_alias(memnode_container, STARPU_POTI_STR_LEN, prefix, node);
		snprintf(paje_value, sizeof(paje_value), "%lx", handle);
		poti_NewEvent(get_event_time_stamp(ev, options), memnode_container, event, paje_value);
#else
		fprintf(out_paje_file, "9	%.9f	%s	%smm%u	%lx\n", get_event_time_stamp(ev, options), event, prefix, node, handle);
#endif
	}
}

static void handle_mpi_data_set_rank(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	unsigned long rank = ev->param[1];
	struct data_info *data = get_data(handle, options->file_rank);

	data->mpi_owner = rank;
}

static void handle_mpi_data_set_tag(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long handle = ev->param[0];
	long tag = ev->param[1];
	struct data_info *data = get_data(handle, options->file_rank);

	data->mpi_tag = tag;
}

static const char *copy_link_type(enum _starpu_is_prefetch prefetch)
{
	switch (prefetch)
	{
		case STARPU_FETCH: return "F";
		case STARPU_PREFETCH: return "PF";
		case STARPU_IDLEFETCH: return "IF";
		default: STARPU_ASSERT(0);
	}
}

static void handle_start_driver_copy(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned src = ev->param[0];
	unsigned dst = ev->param[1];
	unsigned size = ev->param[2];
	unsigned comid = ev->param[3];
	enum _starpu_is_prefetch prefetch = ev->param[4];
	unsigned long handle = ev->param[5];
	const char *link_type = copy_link_type(prefetch);

	char *prefix = options->file_prefix;

	if (!options->no_bus)
	{
		if (out_paje_file)
		{
			double time = get_event_time_stamp(ev, options);
			memnode_push_state(time, prefix, dst, "Co");
			memnode_event(get_event_time_stamp(ev, options), options->file_prefix, dst, "DCo", handle, comid, size, src, options);
#ifdef STARPU_HAVE_POTI
			char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN], src_memnode_container[STARPU_POTI_STR_LEN];
			char program_container[STARPU_POTI_STR_LEN];
			snprintf(paje_value, sizeof(paje_value), "%u", size);
			snprintf(paje_key, sizeof(paje_key), "com_%u", comid);
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
		com->size = size;

		com->src_node = src;
		com->dst_node = dst;
		com->type = link_type;
		com->peer = NULL;
		com->handle = handle;

		_starpu_communication_list_push_back(&communication_list, com);
	}

}


static void handle_work_stealing(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	if (out_paje_file)
	{
		unsigned dst = ev->param[0];
		unsigned src = ev->param[1];
		char *prefix = options->file_prefix;
		unsigned size = 0;
		double time = get_event_time_stamp(ev, options);
#ifdef STARPU_HAVE_POTI
		char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN], src_worker_container[STARPU_POTI_STR_LEN], dst_worker_container[STARPU_POTI_STR_LEN];
		char program_container[STARPU_POTI_STR_LEN];

		snprintf(paje_value, sizeof(paje_value), "%u", size);
		snprintf(paje_key, sizeof(paje_key), "steal_%u", steal_number);
		program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
		worker_container_alias(src_worker_container, STARPU_POTI_STR_LEN, prefix, src);
		worker_container_alias(dst_worker_container, STARPU_POTI_STR_LEN, prefix, dst);
		poti_StartLink(time, program_container, "WSL", src_worker_container, paje_value, paje_key);
		poti_EndLink(time+0.000000001, program_container, "WSL", dst_worker_container, paje_value, paje_key);
#else

		fprintf(out_paje_file, "18	%.9f	WSL	%sp	%u	%sw%u	steal_%u\n", time, prefix, size, prefix, src, steal_number);
		fprintf(out_paje_file, "19	%.9f	WSL	%sp	%u	%sw%u	steal_%u\n", time+0.000000001, prefix, size, prefix, dst, steal_number);
#endif
	}

	steal_number++;
}


static void handle_end_driver_copy(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int src = -1;
	unsigned long handle = 0;
	unsigned dst = ev->param[1];
	unsigned long size = ev->param[2];
	unsigned comid = ev->param[3];
	enum _starpu_is_prefetch prefetch = ev->param[4];
	const char *link_type = copy_link_type(prefetch);

	char *prefix = options->file_prefix;

	if (!options->no_bus)
	{
		/* look for a data transfer to match */
#ifdef STARPU_DEVEL
#warning FIXME: use hash table instead
#endif
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
				com->size = size;

				src = com->src_node = itor->src_node;
				com->dst_node = itor->dst_node;
				com->type = itor->type;
				link_type = itor->type;
				handle = itor->handle;
				com->peer = itor;
				itor->peer = com;

				_starpu_communication_list_push_back(&communication_list, com);

				break;
			}
		}

		if (out_paje_file)
		{
			double time = get_event_time_stamp(ev, options);
			memnode_pop_state(time, prefix, dst);
			memnode_event(get_event_time_stamp(ev, options), options->file_prefix, dst, "DCoE", handle, comid, size, src, options);
#ifdef STARPU_HAVE_POTI
			char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN];
			char dst_memnode_container[STARPU_POTI_STR_LEN], program_container[STARPU_POTI_STR_LEN];
			snprintf(paje_value, sizeof(paje_value), "%lu", size);
			snprintf(paje_key, sizeof(paje_key), "com_%u", comid);
			program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
			memmanager_container_alias(dst_memnode_container, STARPU_POTI_STR_LEN, prefix, dst);
			poti_EndLink(time, program_container, link_type, dst_memnode_container, paje_value, paje_key);
#else
			fprintf(out_paje_file, "19	%.9f	%s	%sp	%lu	%smm%u	com_%u\n", time, link_type, prefix, size, prefix, dst, comid);
#endif
		}
	}
}

static void handle_start_driver_copy_async(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned src = ev->param[0];
	unsigned dst = ev->param[1];

	char *prefix = options->file_prefix;

	if (!options->no_bus)
		if (out_paje_file)
		{
			memnode_push_state(get_event_time_stamp(ev, options), prefix, dst, "CoA");
			memnode_event(get_event_time_stamp(ev, options), options->file_prefix, dst, "DCoA", 0, 0, 0, src, options);
		}

}

static void handle_end_driver_copy_async(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned src = ev->param[0];
	unsigned dst = ev->param[1];

	char *prefix = options->file_prefix;

	if (!options->no_bus)
		if (out_paje_file)
		{
			memnode_pop_state(get_event_time_stamp(ev, options), prefix, dst);
			memnode_event(get_event_time_stamp(ev, options), options->file_prefix, dst, "DCoAE", 0, 0, 0, src, options);
		}
}

/* Currently unused */
STARPU_ATTRIBUTE_UNUSED
static void handle_memnode_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];

	if (out_paje_file)
		memnode_set_state(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr);
}

static void handle_memnode_event_start_3(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];
	unsigned size = ev->param[2];
	unsigned long handle = ev->param[3];

	memnode_event(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr, handle, 0, size, memnode, options);
}

static void handle_memnode_event_start_4(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];
	unsigned dest = ev->param[1];
	if(strcmp(eventstr, "rc")==0)
	{
		//If it is a Request Create, use dest normally
	}
	else
	{
		dest = memnode;
	}
	unsigned size = ev->param[2];
	unsigned long handle = ev->param[3];
	unsigned prefe = ev->param[4];

	memnode_event(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr, handle, prefe, size, dest, options);
}

static void handle_memnode_event_end_3(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];
	unsigned long handle = ev->param[2];
	unsigned info = ev->param[3];

	memnode_event(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr, handle, info, 0, memnode, options);
}

static void handle_memnode_event_start_2(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];
	unsigned long handle = ev->param[2];

	memnode_event(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr, handle, 0, 0, memnode, options);
}

static void handle_memnode_event_end_2(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];
	unsigned long handle = ev->param[2];

	memnode_event(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr, handle, 0, 0, memnode, options);
}

static void handle_push_memnode_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, const char *eventstr)
{
	unsigned memnode = ev->param[0];

	if (out_paje_file)
		memnode_push_state(get_event_time_stamp(ev, options), options->file_prefix, memnode, eventstr);
}

static void handle_pop_memnode_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned memnode = ev->param[0];

	if (out_paje_file)
		memnode_pop_state(get_event_time_stamp(ev, options), options->file_prefix, memnode);
}

static void handle_used_mem(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned memnode = ev->param[0];

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char memnode_container[STARPU_POTI_STR_LEN];
		memmanager_container_alias(memnode_container, STARPU_POTI_STR_LEN, options->file_prefix, memnode);
		poti_SetVariable(get_event_time_stamp(ev, options), memnode_container, "use", (double)ev->param[1] / (1<<20));
#else
		fprintf(out_paje_file, "13	%.9f	%smm%u	use	%f\n",
			get_event_time_stamp(ev, options), options->file_prefix, memnode, (double)ev->param[1] / (1<<20));
#endif
	}
}

static void handle_task_submit_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options, unsigned long tid, const char *eventstr)
{
	char *prefix = options->file_prefix;
	int workerid = find_worker_id(prefixTOnodeid(prefix), tid);
	double timestamp = get_event_time_stamp(ev, options);

	if (workerid >= 0)
	{
		/* Normal worker */
		if (eventstr)
		{
			thread_push_state(timestamp, prefix, tid, eventstr);
			recfmt_thread_push_state(timestamp, prefixTOnodeid(prefix), tid, eventstr, "Runtime");
		}
		else
		{
			thread_pop_state(timestamp, prefix, tid);
			recfmt_thread_pop_state(timestamp, prefixTOnodeid(prefix), tid);
		}
	}
	else if (workerid == -2)
	{
		/* MPI thread */
		if (eventstr)
		{
			mpicommthread_push_state(timestamp, prefix, eventstr);
			recfmt_mpicommthread_push_state(get_event_time_stamp(ev, options), eventstr);
		}
		else
		{
			mpicommthread_pop_state(timestamp, prefix);
			recfmt_mpicommthread_pop_state(get_event_time_stamp(ev, options));
		}
	}
	else
	{
		if (eventstr)
		{
			user_thread_push_state(timestamp, prefix, tid, eventstr);
			recfmt_user_thread_push_state(timestamp, tid, eventstr, "User");
		}
		else
		{
			user_thread_pop_state(timestamp, prefix, tid);
			recfmt_user_thread_pop_state(timestamp, tid);
		}
	}
}

/*
 *	Number of task submitted to the scheduler
 */
static int curq_size = 0;
static int nsubmitted = 0;

static void handle_job_push(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double current_timestamp = get_event_time_stamp(ev, options);

	unsigned task = ev->param[0];

	curq_size++;

	_starpu_fxt_component_update_ntasks(nsubmitted, curq_size);

	if (!options->no_counter && out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];

		scheduler_container_alias(container, STARPU_POTI_STR_LEN, options->file_prefix);
		poti_SetVariable(current_timestamp, container, "nready", (double)curq_size);

               char paje_value[STARPU_POTI_STR_LEN];
               snprintf(paje_value, sizeof(paje_value), "%u", task);
               snprintf(container, sizeof(container), "%sp", options->file_prefix);
		if (!options->no_events)
			poti_NewEvent(get_event_time_stamp(ev, options), container, "pu", paje_value);
#else
		fprintf(out_paje_file, "13	%.9f	%ssched	nready	%f\n", current_timestamp, options->file_prefix, (float)curq_size);
		if (!options->no_events)
			fprintf(out_paje_file, "9       %.9f    %s      %sp     %u\n", get_event_time_stamp(ev, options), "pu", options->file_prefix, task);
#endif
	}

	if (activity_file)
		fprintf(activity_file, "cnt_ready\t%.9f\t%d\n", current_timestamp, curq_size);
}


static void handle_job_pop(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double current_timestamp = get_event_time_stamp(ev, options);
	unsigned task = ev->param[0];

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

		char paje_value[STARPU_POTI_STR_LEN];
		snprintf(paje_value, sizeof(paje_value), "%u", task);
		snprintf(container, sizeof(container), "%sp", options->file_prefix);
		if (!options->no_events)
			poti_NewEvent(get_event_time_stamp(ev, options), container, "po", paje_value);
#else
		fprintf(out_paje_file, "13	%.9f	%ssched	nready	%f\n", current_timestamp, options->file_prefix, (float)curq_size);
		fprintf(out_paje_file, "13	%.9f	%ssched	nsubmitted	%f\n", current_timestamp, options->file_prefix, (float)nsubmitted);
		if (!options->no_events)
			fprintf(out_paje_file, "9       %.9f    %s      %sp     %u\n", get_event_time_stamp(ev, options), "po", options->file_prefix, task);
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
	_starpu_fxt_component_new(ev->param[0], get_fxt_string(ev, 1));
}

static void handle_component_connect(struct fxt_ev_64 *ev, struct starpu_fxt_options *options STARPU_ATTRIBUTE_UNUSED)
{
	_starpu_fxt_component_connect(ev->param[0], ev->param[1]);
}

static void handle_component_push(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	double current_timestamp = get_event_time_stamp(ev, options);
	int workerid = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
	_starpu_fxt_component_push(anim_file, options, current_timestamp, workerid, ev->param[1], ev->param[2], ev->param[3], ev->param[4]);
}

static void handle_component_pull(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;
	double current_timestamp = get_event_time_stamp(ev, options);
	int workerid = find_worker_id(prefixTOnodeid(prefix), ev->param[0]);
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

static void handle_tag(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	uint64_t tag;
	unsigned long job;

	tag = ev->param[0];
	job = ev->param[1];

	if (options->label_deps)
		_starpu_fxt_dag_add_tag(options->file_prefix, tag, job, "tag");
	else
		_starpu_fxt_dag_add_tag(options->file_prefix, tag, job, NULL);
}

static void handle_tag_deps(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	uint64_t child;
	uint64_t father;

	child = ev->param[0];
	father = ev->param[1];

	if (options->label_deps)
		_starpu_fxt_dag_add_tag_deps(options->file_prefix, child, father, "tag");
	else
		_starpu_fxt_dag_add_tag_deps(options->file_prefix, child, father, NULL);
}

static void handle_task_deps(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long dep_prev = ev->param[0];
	unsigned long dep_succ = ev->param[1];
	unsigned dep_succ_type = ev->param[2];
	char *name = get_fxt_string(ev,4);

	struct task_info *task = get_task(dep_succ, options->file_rank);
	struct task_info *prev_task = get_task(dep_prev, options->file_rank);
	unsigned alloc = 0;

	task->type = dep_succ_type;

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
	{
		_STARPU_REALLOC(task->dependencies, sizeof(*task->dependencies) * alloc);
		_STARPU_REALLOC(task->dep_labels, sizeof(*task->dep_labels) * alloc);
	}
	task->dependencies[task->ndeps] = dep_prev;
	task->dep_labels[task->ndeps] = strdup(name);
	task->ndeps++;

	/* There is a dependency between both job id : dep_prev -> dep_succ */
	if (show_task(task, options) && show_task(prev_task, options))
	{
		if (!options->label_deps) name = NULL;
		/* We should show the name of the predecessor, then. */
		prev_task->show = 1;
		_starpu_fxt_dag_add_task_deps(options->file_prefix, dep_prev, dep_succ, name);
	}
}

static void handle_task_submit(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long job_id = ev->param[0];
	unsigned long iteration = ev->param[1];
	unsigned long subiteration = ev->param[2];
	unsigned long submit_order = ev->param[3];
	long priority = (long) ev->param[4];
	unsigned type = ev->param[5];

	struct task_info *task = get_task(job_id, options->file_rank);
	task->submit_time = get_event_time_stamp(ev, options);
	task->submit_order = submit_order;
	task->priority = priority;
	task->iterations[0] = iteration;
	task->iterations[1] = subiteration;
	task->type = type;
}

static void handle_task_color(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long job_id = ev->param[0];
	struct task_info *task = get_task(job_id, options->file_rank);
	int color = (long) ev->param[1];

	task->color = color;
}

static void handle_task_exclude_from_dag(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long job_id = ev->param[0];
	unsigned exclude_from_dag = ev->param[1];

	struct task_info *task = get_task(job_id, options->file_rank);
	task->exclude_from_dag = exclude_from_dag;
}

static void handle_task_name(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	unsigned long job_id = ev->param[0];
	char *name = get_fxt_string(ev,1);

	char *prefix = options->file_prefix;
	struct task_info *task = get_task(job_id, options->file_rank);
        int worker = find_worker_id(prefixTOnodeid(prefix), ev->param[1]);

	const char *color;
	char buffer[32];
	if (task->color != 0)
	{
		snprintf(buffer, sizeof(buffer), "#%06x", task->color);
		color = &buffer[0];
	}
	else if (options->per_task_colour)
	{
		snprintf(buffer, sizeof(buffer), "#%x%x%x",
			 get_color_symbol_red(name)/4,
			 get_color_symbol_green(name)/4,
			 get_color_symbol_blue(name)/4);
		color = &buffer[0];
	}
	else
	{
		color= (worker < 0)?"#aaaaaa":get_worker_color(worker);
	}

	if (!task->name)
		task->name = strdup(name);

	if (!task->exclude_from_dag && show_task(task, options))
		_starpu_fxt_dag_set_task_name(options->file_prefix, job_id, task->name, color);
}

static void handle_task_done(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
        /* Ideally, we would be able to dump tasks as they terminate, to save
         * memory.
         * We however may have to change their state later, e.g. the show field,
         * due to dependencies added way later. */
#if 0
	unsigned long job_id;
	job_id = ev->param[0];

	struct task_info *task = get_task(job_id, options->file_rank);

	task_dump(task, options);
#else
	(void) ev;
	(void) options;
#endif
}

static void handle_tag_done(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;

	uint64_t tag_id;
	tag_id = ev->param[0];

	unsigned long has_name = ev->param[2];
	char *name = has_name?get_fxt_string(ev,3):"unknown";

        int worker;
        worker = find_worker_id(prefixTOnodeid(prefix), ev->param[1]);

	const char *color;
	char buffer[32];
	if (options->per_task_colour)
	{
		snprintf(buffer, sizeof(buffer), "%.4f,%.4f,%.4f",
			 get_color_symbol_red(name)/1024.0,
			 get_color_symbol_green(name)/1024.0,
			 get_color_symbol_blue(name)/1024.0);
		color = &buffer[0];
	}
	else
	{
		color= (worker < 0)?"white":get_worker_color(worker);
	}

	_starpu_fxt_dag_set_tag_done(options->file_prefix, tag_id, color);
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
		snprintf(container, sizeof(container), "%sp", options->file_prefix);
		snprintf(paje_value, sizeof(paje_value), "%d", rank);
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

	register_mpi_thread(prefixTOnodeid(prefix), ev->param[2]);

	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char program_container[STARPU_POTI_STR_LEN];
		program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
		char new_mpicommthread_container_alias[STARPU_POTI_STR_LEN];
		mpicommthread_container_alias(new_mpicommthread_container_alias, STARPU_POTI_STR_LEN, prefix);
		snprintf(new_mpicommthread_container_alias, STARPU_POTI_STR_LEN, "%smpict", prefix);
		poti_CreateContainer(date, new_mpicommthread_container_alias, "MPICt", program_container, new_mpicommthread_container_alias);
		//set bandwidth variables to zero when they start
		poti_SetVariable(date, new_mpicommthread_container_alias, "bwi_mpi", 0.);
		poti_SetVariable(date, new_mpicommthread_container_alias, "bwo_mpi", 0.);
#else
		fprintf(out_paje_file, "7	%.9f	%smpict		MPICt	%sp	%smpict\n", date, prefix, prefix, prefix);
		//set bandwidth variables to zero when they start
		fprintf(out_paje_file, "13	%.9f	%smpict	bwi_mpi	0.0\n", date, prefix);
		fprintf(out_paje_file, "13	%.9f	%smpict	bwo_mpi	0.0\n", date, prefix);
#endif
		mpicommthread_set_state(date, prefix, "Sl");
	}
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "Sl");

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
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "SdS");
}

static int mpi_warned;
static void handle_mpi_isend_submit_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int dest = ev->param[0];
	int mpi_tag = ev->param[1];
	size_t size = ev->param[2];
	long jobid = ev->param[3];
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "P");

	if (options->file_rank < 0)
	{
		if (!mpi_warned)
		{
			_STARPU_MSG("Warning : Only one trace file is given. MPI transfers will not be displayed. Add all trace files to show them ! \n");
			mpi_warned = 1;
		}
	}
	else
		_starpu_fxt_mpi_add_send_transfer(options->file_rank, dest, mpi_tag, size, date, jobid);
}

static void handle_mpi_irecv_submit_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "RvS");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "RvS");
}

static void handle_mpi_irecv_submit_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "P");
}

static void handle_mpi_isend_complete_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "SdC");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "SdC");
}

static void handle_mpi_isend_complete_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "P");
}

static void handle_mpi_irecv_complete_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "RvC");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "RvC");
}

static void handle_mpi_irecv_complete_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "P");
}

static void handle_mpi_irecv_terminated(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	int src = ev->param[0];
	int mpi_tag = ev->param[1];
	long jobid = ev->param[2];
	double date = get_event_time_stamp(ev, options);

	if (options->file_rank < 0)
	{
		if (!mpi_warned)
		{
			_STARPU_MSG("Warning : Only one trace file is given. MPI transfers will not be displayed. Add all trace files to show them ! \n");
			mpi_warned = 1;
		}
	}
	else
		_starpu_fxt_mpi_add_recv_transfer(src, options->file_rank, mpi_tag, date, jobid);
}

static void handle_mpi_sleep_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "Sl");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "Sl");
}

static void handle_mpi_sleep_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "Pl");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "Pl");
}

static void handle_mpi_dtesting_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "DT");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "DT");
}

static void handle_mpi_dtesting_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "P");
}

static void handle_mpi_utesting_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "UT");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "UT");
}

static void handle_mpi_utesting_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "P");
}

static void handle_mpi_uwait_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "UW");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "UW");
}

static void handle_mpi_uwait_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
	if (trace_file)
		recfmt_mpicommthread_set_state(date, "P");
}

static void handle_mpi_testing_detached_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_push_state(date, options->file_prefix, "TD");
	if (trace_file)
		recfmt_mpicommthread_push_state(date, "TD");
}

static void handle_mpi_testing_detached_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_pop_state(date, options->file_prefix);
	if (trace_file)
		recfmt_mpicommthread_pop_state(date);
}

static void handle_mpi_test_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_push_state(date, options->file_prefix, "MT");
	if (trace_file)
		recfmt_mpicommthread_push_state(date, "MT");
}

static void handle_mpi_test_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_pop_state(date, options->file_prefix);
	if (trace_file)
		recfmt_mpicommthread_pop_state(date);
}

static void handle_mpi_polling_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "Pl");
}

static void handle_mpi_polling_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "P");
}

static void handle_mpi_driver_run_begin(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "Dr");
}

static void handle_mpi_driver_run_end(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	double date = get_event_time_stamp(ev, options);

	if (out_paje_file)
		mpicommthread_set_state(date, options->file_prefix, "Pl");
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

static void handle_string_event(struct fxt_ev_64 *ev, const char *event, struct starpu_fxt_options *options)
{
	/* Add an event in the trace */
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];
		snprintf(container, sizeof(container), "%sp", options->file_prefix);
		poti_NewEvent(get_event_time_stamp(ev, options), container, "prog_event", event);
#else
		fprintf(out_paje_file, "9	%.9f	prog_event	%sp	\"%s\"\n", get_event_time_stamp(ev, options), options->file_prefix, event);
#endif
	}

	if (trace_file)
		recfmt_dump_state(get_event_time_stamp(ev, options), "ProgEvent", -1, 0, event, "Program");
}

static void handle_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	char *event = get_fxt_string(ev, 0);
	handle_string_event(ev, event, options);
}

static void handle_thread_event(struct fxt_ev_64 *ev, struct starpu_fxt_options *options)
{
	/* Add an event in the trace */
	if (out_paje_file)
	{
		char *event = get_fxt_string(ev, 1);

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
void _starpu_fxt_process_bandwidth(struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;

	/* Loop through completed communications */
	while (!_starpu_communication_list_empty(&communication_list)
			&& _starpu_communication_list_begin(&communication_list)->peer)
	{
		struct _starpu_communication*itor;
		/* This communication is complete */
		itor = _starpu_communication_list_pop_front(&communication_list);

		current_bandwidth_out_per_node[itor->src_node] +=  itor->bandwidth;
		if (out_paje_file)
		{
#ifdef STARPU_HAVE_POTI
			char src_memnode_container[STARPU_POTI_STR_LEN];
			memmanager_container_alias(src_memnode_container, STARPU_POTI_STR_LEN, prefix, itor->src_node);
			poti_SetVariable(itor->comm_start, src_memnode_container, "bwo_mm", current_bandwidth_out_per_node[itor->src_node]);
#else
			fprintf(out_paje_file, "13	%.9f	%smm%u	bwo_mm	%f\n",
				itor->comm_start, prefix, itor->src_node, current_bandwidth_out_per_node[itor->src_node]);
#endif
		}

		current_bandwidth_in_per_node[itor->dst_node] +=  itor->bandwidth;
		if (out_paje_file)
		{
#ifdef STARPU_HAVE_POTI
			char dst_memnode_container[STARPU_POTI_STR_LEN];
			memmanager_container_alias(dst_memnode_container, STARPU_POTI_STR_LEN, prefix, itor->dst_node);
			poti_SetVariable(itor->comm_start, dst_memnode_container, "bwi_mm", current_bandwidth_in_per_node[itor->dst_node]);
#else
			fprintf(out_paje_file, "13	%.9f	%smm%u	bwi_mm	%f\n",
				itor->comm_start, prefix, itor->dst_node, current_bandwidth_in_per_node[itor->dst_node]);
#endif
		}
		_starpu_communication_delete(itor);
	}
}

static
void _starpu_fxt_process_computations(struct starpu_fxt_options *options)
{
	char *prefix = options->file_prefix;

	/* Loop through completed computations */
	struct _starpu_computation*itor;
	while (!_starpu_computation_list_empty(&computation_list)
			&& _starpu_computation_list_begin(&computation_list)->peer)
	{
		/* This computation is complete */
		itor = _starpu_computation_list_pop_front(&computation_list);

		if (out_paje_file && itor->comp_start != current_computation_time)
		{
			/* flush last value */
#ifdef STARPU_HAVE_POTI
			char container[STARPU_POTI_STR_LEN];

			scheduler_container_alias(container, STARPU_POTI_STR_LEN, prefix);
			poti_SetVariable(current_computation_time, container, "gft", (double)current_computation);
#else
			fprintf(out_paje_file, "13	%.9f	%ssched	gft	%f\n", current_computation_time, prefix, (float)current_computation);
#endif
		}
		current_computation += itor->gflops;
		current_computation_time = itor->comp_start;

		_starpu_computation_delete(itor);
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
		STARPU_ABORT_MSG("Failed to open '%s' (err %s)", filename_in, strerror(errno));
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

	char *prefix = options->file_prefix;

	/* TODO starttime ...*/
	/* create the "program" container */
	current_computation = 0.0;
	current_computation_time = 0.0;
	if (out_paje_file)
	{
#ifdef STARPU_HAVE_POTI
		char new_program_container_alias[STARPU_POTI_STR_LEN], new_program_container_name[STARPU_POTI_STR_LEN];
		program_container_alias(new_program_container_alias, STARPU_POTI_STR_LEN, prefix);
		snprintf(new_program_container_name, sizeof(new_program_container_name), "program %s", prefix);
		poti_CreateContainer (0, new_program_container_alias, "P", "MPIroot", new_program_container_name);
		char new_scheduler_container_alias[STARPU_POTI_STR_LEN], new_scheduler_container_name[STARPU_POTI_STR_LEN];
		scheduler_container_alias(new_scheduler_container_alias, STARPU_POTI_STR_LEN, prefix);
		snprintf(new_scheduler_container_name, sizeof(new_scheduler_container_name), "%sscheduler", prefix);
		if (!options->no_counter || !options->no_flops)
		{
			poti_CreateContainer(0.0, new_scheduler_container_alias, "Sc", new_program_container_alias, new_scheduler_container_name);
		}
		if (!options->no_counter)
		{
			poti_SetVariable(0.0, new_scheduler_container_alias, "nsubmitted", 0.0);
			poti_SetVariable(0.0, new_scheduler_container_alias, "nready", 0.0);
		}
		if (!options->no_flops)
		{
			poti_SetVariable(0.0, new_scheduler_container_alias, "gft", 0.0);
		}
#else
		fprintf(out_paje_file, "7	0.0	%sp	P	MPIroot	%sprogram \n", prefix, prefix);
		if (!options->no_counter || !options->no_flops)
		{
			fprintf(out_paje_file, "7	0.0	%ssched	Sc	%sp	%sscheduler\n", prefix, prefix, prefix);
		}
		if (!options->no_counter)
		{
		/* create a variable with the number of tasks */
			fprintf(out_paje_file, "13	0.0	%ssched	nsubmitted	0.0\n", prefix);
			fprintf(out_paje_file, "13	0.0	%ssched	nready	0.0\n", prefix);
		}
		if (!options->no_flops)
		{
			fprintf(out_paje_file, "13	0.0	%ssched	gft	0.0\n", prefix);
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
			case _STARPU_FUT_START_FETCH_INPUT_ON_TID:
				handle_worker_status_on_tid(&ev, options, "Fi");
				break;
			case _STARPU_FUT_START_PUSH_OUTPUT_ON_TID:
				handle_worker_status_on_tid(&ev, options, "Po");
				break;
			case _STARPU_FUT_START_PROGRESS_ON_TID:
				handle_worker_status_on_tid(&ev, options, "P");
				break;
			case _STARPU_FUT_START_UNPARTITION_ON_TID:
				handle_worker_status_on_tid(&ev, options, "U");
				break;
			case _STARPU_FUT_END_FETCH_INPUT_ON_TID:
			case _STARPU_FUT_END_PROGRESS_ON_TID:
			case _STARPU_FUT_END_PUSH_OUTPUT_ON_TID:
			case _STARPU_FUT_END_UNPARTITION_ON_TID:
				handle_worker_status_on_tid(&ev, options, "B");
				break;

			case _STARPU_FUT_START_FETCH_INPUT:
				handle_worker_status(&ev, options, "Fi");
				break;

			case _STARPU_FUT_END_FETCH_INPUT:
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
				handle_tag(&ev, options);
				break;

			case _STARPU_FUT_TAG_DEPS:
				handle_tag_deps(&ev, options);
				break;

			case _STARPU_FUT_TASK_DEPS:
				handle_task_deps(&ev, options);
				break;

			case _STARPU_FUT_TASK_SUBMIT:
				handle_task_submit(&ev, options);
				break;

			case _STARPU_FUT_TASK_BUILD_START:
				handle_task_submit_event(&ev, options, ev.param[0], "Bu");
				break;

			case _STARPU_FUT_TASK_SUBMIT_START:
				handle_task_submit_event(&ev, options, ev.param[0], "Su");
				break;

			case _STARPU_FUT_TASK_THROTTLE_START:
				handle_task_submit_event(&ev, options, ev.param[0], "Th");
				break;

			case _STARPU_FUT_TASK_MPI_DECODE_START:
				handle_task_submit_event(&ev, options, ev.param[0], "MD");
				break;

			case _STARPU_FUT_TASK_MPI_PRE_START:
				handle_task_submit_event(&ev, options, ev.param[0], "MPr");
				break;

			case _STARPU_FUT_TASK_MPI_POST_START:
				handle_task_submit_event(&ev, options, ev.param[0], "MPo");
				break;

			case _STARPU_FUT_TASK_WAIT_START:
				handle_task_submit_event(&ev, options, ev.param[1], "W");
				break;

			case _STARPU_FUT_TASK_WAIT_FOR_ALL_START:
				handle_task_submit_event(&ev, options, ev.param[0], "WA");
				break;

			case _STARPU_FUT_TASK_BUILD_END:
			case _STARPU_FUT_TASK_SUBMIT_END:
			case _STARPU_FUT_TASK_THROTTLE_END:
			case _STARPU_FUT_TASK_MPI_DECODE_END:
			case _STARPU_FUT_TASK_MPI_PRE_END:
			case _STARPU_FUT_TASK_MPI_POST_END:
			case _STARPU_FUT_TASK_WAIT_FOR_ALL_END:
				handle_task_submit_event(&ev, options, ev.param[0], NULL);
				break;

			case _STARPU_FUT_TASK_WAIT_END:
				handle_task_submit_event(&ev, options, ev.param[0], NULL);
				break;

			case _STARPU_FUT_TASK_EXCLUDE_FROM_DAG:
				handle_task_exclude_from_dag(&ev, options);
				break;

			case _STARPU_FUT_TASK_NAME:
				handle_task_name(&ev, options);
				break;

			case _STARPU_FUT_TASK_COLOR:
				handle_task_color(&ev, options);
				break;

			case _STARPU_FUT_TASK_DONE:
				handle_task_done(&ev, options);
				break;

			case _STARPU_FUT_TAG_DONE:
				handle_tag_done(&ev, options);
				break;

			case _STARPU_FUT_HANDLE_DATA_REGISTER:
				handle_data_register(&ev, options);
				break;

			case _STARPU_FUT_HANDLE_DATA_UNREGISTER:
				handle_data_unregister(&ev, options);
				break;

			case _STARPU_FUT_DATA_STATE_INVALID:
				if (options->memory_states)
					handle_data_state(&ev, options, "SI");
				break;
			case _STARPU_FUT_DATA_STATE_OWNER:
				if (options->memory_states)
					handle_data_state(&ev, options, "SO");
				break;
			case _STARPU_FUT_DATA_STATE_SHARED:
				if (options->memory_states)
					handle_data_state(&ev, options, "SS");
				break;
                       case _STARPU_FUT_DATA_REQUEST_CREATED:
                               if (!options->no_bus && options->memory_states)
                               {
                                       handle_memnode_event_start_4(&ev, options, "rc");
                               }
                               break;
			case _STARPU_FUT_DATA_COPY:
				if (!options->no_bus)
				     handle_data_copy();
				break;

			case _STARPU_FUT_DATA_LOAD:
			     	break;

			case _STARPU_FUT_DATA_NAME:
				handle_data_name(&ev, options);
				break;

			case _STARPU_FUT_DATA_COORDINATES:
				handle_data_coordinates(&ev, options);
				break;

			case _STARPU_FUT_DATA_WONT_USE:
				handle_data_wont_use(&ev, options);
				break;

			case _STARPU_FUT_DATA_DOING_WONT_USE:
				if (options->memory_states)
					handle_data_doing_wont_use(&ev, options);
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
				{
					handle_push_memnode_event(&ev, options, "A");
                                       handle_memnode_event_start_4(&ev, options, "Al");
				}
				break;
			case _STARPU_FUT_START_ALLOC_REUSE:
				if (!options->no_bus)
				{
					handle_push_memnode_event(&ev, options, "Ar");
                                       handle_memnode_event_start_4(&ev, options, "Alr");
				}
				break;
			case _STARPU_FUT_END_ALLOC:
				if (!options->no_bus)
				{
					handle_pop_memnode_event(&ev, options);
					handle_memnode_event_end_3(&ev, options, "AlE");
				}
				break;
			case _STARPU_FUT_END_ALLOC_REUSE:
				if (!options->no_bus)
				{
					handle_pop_memnode_event(&ev, options);
					handle_memnode_event_end_3(&ev, options, "AlrE");
				}
				break;
			case _STARPU_FUT_START_FREE:
				if (!options->no_bus)
				{
					handle_push_memnode_event(&ev, options, "F");
					handle_memnode_event_start_3(&ev, options, "Fe");
				}
				break;
			case _STARPU_FUT_END_FREE:
				if (!options->no_bus)
				{
					handle_pop_memnode_event(&ev, options);
					handle_memnode_event_end_2(&ev, options, "FeE");
				}
				break;
			case _STARPU_FUT_START_WRITEBACK:
				if (!options->no_bus)
				{
					handle_push_memnode_event(&ev, options, "W");
					handle_memnode_event_start_2(&ev, options, "Wb");
				}
				break;
			case _STARPU_FUT_END_WRITEBACK:
				if (!options->no_bus)
				{
					handle_pop_memnode_event(&ev, options);
					handle_memnode_event_start_2(&ev, options, "WbE");
				}
				break;
			case _STARPU_FUT_START_WRITEBACK_ASYNC:
				if (!options->no_bus)
					handle_push_memnode_event(&ev, options, "Wa");
				break;
			case _STARPU_FUT_END_WRITEBACK_ASYNC:
				if (!options->no_bus)
					handle_pop_memnode_event(&ev, options);
				break;
			case _STARPU_FUT_START_MEMRECLAIM:
				if (!options->no_bus)
					handle_push_memnode_event(&ev, options, "R");
				break;
			case _STARPU_FUT_END_MEMRECLAIM:
				if (!options->no_bus)
					handle_pop_memnode_event(&ev, options);
				break;
			case _STARPU_FUT_USED_MEM:
				handle_used_mem(&ev, options);
				break;

			case _STARPU_FUT_USER_EVENT:
				if (!options->no_events)
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

			case _STARPU_MPI_FUT_ISEND_TERMINATED:
				break;

			case _STARPU_MPI_FUT_IRECV_TERMINATED:
				handle_mpi_irecv_terminated(&ev, options);
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

			case _STARPU_MPI_FUT_DATA_SET_RANK:
				handle_mpi_data_set_rank(&ev, options);
				break;
			case _STARPU_MPI_FUT_DATA_SET_TAG:
				handle_mpi_data_set_tag(&ev, options);
				break;

			case _STARPU_MPI_FUT_TESTING_DETACHED_BEGIN:
				handle_mpi_testing_detached_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_TESTING_DETACHED_END:
				handle_mpi_testing_detached_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_TEST_BEGIN:
				handle_mpi_test_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_TEST_END:
				handle_mpi_test_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_POLLING_BEGIN:
				handle_mpi_polling_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_POLLING_END:
				handle_mpi_polling_end(&ev, options);
				break;

			case _STARPU_MPI_FUT_DRIVER_RUN_BEGIN:
				handle_mpi_driver_run_begin(&ev, options);
				break;

			case _STARPU_MPI_FUT_DRIVER_RUN_END:
				handle_mpi_driver_run_end(&ev, options);
				break;

			case _STARPU_FUT_SET_PROFILING:
				handle_set_profiling(&ev, options);
				break;

			case _STARPU_FUT_TASK_WAIT_FOR_ALL:
				handle_task_wait_for_all();
				break;

			case _STARPU_FUT_EVENT:
				if (!options->no_events)
					handle_event(&ev, options);
				break;

			case _STARPU_FUT_THREAD_EVENT:
				if (!options->no_events)
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

			case FUT_SETUP_CODE:
				fut_keymask = ev.param[0];
				break;

			case FUT_KEYCHANGE_CODE:
				fut_keymask = ev.param[0];
				break;

			case FUT_START_FLUSH_CODE:
				handle_string_event(&ev, "fxt_start_flush", options);
				break;
			case FUT_STOP_FLUSH_CODE:
				handle_string_event(&ev, "fxt_stop_flush", options);
				break;

			/* We can safely ignore FUT internal events */
			case FUT_CALIBRATE0_CODE:
			case FUT_CALIBRATE1_CODE:
			case FUT_CALIBRATE2_CODE:
			case FUT_NEW_LWP_CODE:
			case FUT_GCC_INSTRUMENT_ENTRY_CODE:
				break;

			default:
#ifdef STARPU_VERBOSE
				_STARPU_MSG("unknown event.. %x at time %llx WITH OFFSET %llx\n",
					    (unsigned)ev.code, (long long unsigned)ev.time, (long long unsigned)(ev.time-options->file_offset));
#endif
				break;
		}
		_starpu_fxt_process_bandwidth(options);
		if (!options->no_flops)
			_starpu_fxt_process_computations(options);
	}

	if (!options->no_flops)
	{
		/* computations are supposed to be over, drop any pending comp */
		unsigned i;
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			struct _starpu_computation *comp = ongoing_computation[i];
			if (comp)
			{
				STARPU_ASSERT(!comp->peer);
				_starpu_computation_list_erase(&computation_list, comp);
				ongoing_computation[i] = 0;
			}
		}
		/* And flush completed computations */
		_starpu_fxt_process_computations(options);
	}

	if (out_paje_file && !options->no_bus)
	{
		while (!_starpu_communication_list_empty(&communication_list)) {
			struct _starpu_communication*itor;
			itor = _starpu_communication_list_pop_front(&communication_list);

			/* Trace finished with this communication uncompleted, fake its termination */

			unsigned comid = itor->comid;
			unsigned long size = itor->size;
			unsigned dst = itor->dst_node;
			double time = current_computation_time;
			const char *link_type = itor->type;
#ifdef STARPU_HAVE_POTI
			char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN];
			char dst_memnode_container[STARPU_POTI_STR_LEN], program_container[STARPU_POTI_STR_LEN];
			snprintf(paje_value, sizeof(paje_value), "%lu", size);
			snprintf(paje_key, sizeof(paje_key), "com_%u", comid);
			program_container_alias(program_container, STARPU_POTI_STR_LEN, prefix);
			memmanager_container_alias(dst_memnode_container, STARPU_POTI_STR_LEN, prefix, dst);
			poti_EndLink(time, program_container, link_type, dst_memnode_container, paje_value, paje_key);
#else
			fprintf(out_paje_file, "19	%.9f	%s	%sp	%lu	%smm%u	com_%u\n", time, link_type, prefix, size, prefix, dst, comid);
#endif
			_starpu_communication_delete(itor);
		}
	}

	if (out_paje_file && !options->no_flops)
	{
		unsigned i;
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (last_codelet_end[i] != 0.0)
			{
#ifdef STARPU_HAVE_POTI
				char container[STARPU_POTI_STR_LEN];
				worker_container_alias(container, STARPU_POTI_STR_LEN, prefix, i);
				poti_SetVariable(last_codelet_end[i], container, "gf", 0.);
#else
				fprintf(out_paje_file, "13	%.9f	%sw%u	gf	%f\n",
						last_codelet_end[i], prefix, i, 0.);
#endif
				last_codelet_end[i] = 0.0;
			}
		}

		/* flush last value */
#ifdef STARPU_HAVE_POTI
		char container[STARPU_POTI_STR_LEN];

		scheduler_container_alias(container, STARPU_POTI_STR_LEN, prefix);
		poti_SetVariable(current_computation_time, container, "gft", (double)current_computation);
#else
		fprintf(out_paje_file, "13	%.9f	%ssched	gft	%f\n", current_computation_time, prefix, (float)current_computation);
#endif
	}

	{
		struct data_info *data=NULL, *tmp=NULL;
		HASH_ITER(hh, data_info, data, tmp)
		{
			data_dump(data);
		}
	}

	{
		struct task_info *task=NULL, *tmp=NULL;
		HASH_ITER(hh, tasks_info, task, tmp)
		{
			task_dump(task, options);
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
	options->no_events = 0;
	options->no_counter = 0;
	options->no_bus = 0;
	options->no_flops = 0;
	options->no_smooth = 0;
	options->no_acquire = 0;
	options->memory_states = 0;
	options->internal = 0;
	options->label_deps = 0;
	options->ninputfiles = 0;
	options->out_paje_path = "paje.trace";
	options->dag_path = "dag.dot";
	options->tasks_path = "tasks.rec";
	options->data_path = "data.rec";
	options->anim_path = "trace.html";
	options->states_path = "trace.rec";
	options->distrib_time_path = "distrib.data";
	options->dumped_codelets = NULL;
	options->activity_path = "activity.data";
}

void _set_dir(char *dir, char **option)
{
	if (*option)
	{
		char *tmp = strdup(*option);
		_STARPU_MALLOC(*option, 256);
		snprintf(*option, 256, "%s/%s", dir, tmp);
		free(tmp);
	}
}

void starpu_fxt_options_set_dir(struct starpu_fxt_options *options)
{
	if (!options->dir)
		return;

	_starpu_mkpath_and_check(options->dir, S_IRWXU);
	_set_dir(options->dir, &options->out_paje_path);
	_set_dir(options->dir, &options->dag_path);
	_set_dir(options->dir, &options->tasks_path);
	_set_dir(options->dir, &options->data_path);
	_set_dir(options->dir, &options->anim_path);
	_set_dir(options->dir, &options->states_path);
	_set_dir(options->dir, &options->distrib_time_path);
	_set_dir(options->dir, &options->activity_path);
}

void starpu_fxt_options_shutdown(struct starpu_fxt_options *options)
{
	if (options->dir)
	{
		free(options->out_paje_path);
		free(options->dag_path);
		free(options->tasks_path);
		free(options->data_path);
		free(options->anim_path);
		free(options->states_path);
		free(options->distrib_time_path);
		free(options->activity_path);
	}
}

static
void _starpu_fxt_distrib_file_init(struct starpu_fxt_options *options)
{
	dumped_codelets_count = 0;
	dumped_codelets = NULL;

	if (options->distrib_time_path)
	{
		distrib_time = fopen(options->distrib_time_path, "w+");
		if (distrib_time == NULL)
			STARPU_ABORT_MSG("Failed to open '%s' (err %s)", options->distrib_time_path, strerror(errno));
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
	{
		activity_file = fopen(options->activity_path, "w+");
		if (activity_file == NULL)
			STARPU_ABORT_MSG("Failed to open '%s' (err %s)", options->activity_path, strerror(errno));
	}
	else
		activity_file = NULL;
}

static
void _starpu_fxt_anim_file_init(struct starpu_fxt_options *options)
{
	if (options->anim_path)
	{
		anim_file = fopen(options->anim_path, "w+");
		if (anim_file == NULL)
			STARPU_ABORT_MSG("Failed to open '%s' (err %s)", options->anim_path, strerror(errno));

		_starpu_fxt_component_print_header(anim_file);
	}
	else
		anim_file = NULL;
}

static
void _starpu_fxt_tasks_file_init(struct starpu_fxt_options *options)
{
	if (options->tasks_path)
	{
		tasks_file = fopen(options->tasks_path, "w+");
		if (tasks_file == NULL)
			STARPU_ABORT_MSG("Failed to open '%s' (err %s)", options->tasks_path, strerror(errno));
	}
	else
		tasks_file = NULL;
}

static
void _starpu_fxt_data_file_init(struct starpu_fxt_options *options)
{
	if (options->data_path)
	{
		data_file = fopen(options->data_path, "w+");
		if (data_file == NULL)
			STARPU_ABORT_MSG("Failed to open '%s' (err %s)", options->data_path, strerror(errno));
	}
	else
		data_file = NULL;
}

static
void _starpu_fxt_write_trace_header(FILE *f)
{
	fprintf(f, "#\n");
	fprintf(f, "# E: Event type\n");
	fprintf(f, "# N: Event name\n");
	fprintf(f, "# C: Event category\n");
	fprintf(f, "# W: Worker ID\n");
	fprintf(f, "# T: Thread ID\n");
	fprintf(f, "# S: Start time\n");
	fprintf(f, "#\n");
	fprintf(f, "\n");
}

static
void _starpu_fxt_trace_file_init(struct starpu_fxt_options *options)
{
	if (options->states_path)
	{
		trace_file = fopen(options->states_path, "w+");
		if (trace_file == NULL)
			STARPU_ABORT_MSG("Failed to open '%s' (err %s)", options->states_path, strerror(errno));
	}
	else
		trace_file = NULL;

	if (trace_file)
		_starpu_fxt_write_trace_header(trace_file);
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
	if (anim_file)
	{
		_starpu_fxt_component_finish(anim_file);
		fclose(anim_file);
	}
}

static
void _starpu_fxt_tasks_file_close(void)
{
	if (tasks_file)
		fclose(tasks_file);
}

static
void _starpu_fxt_data_file_close(void)
{
	if (data_file)
		fclose(data_file);
}

static
void _starpu_fxt_trace_file_close(void)
{
	if (trace_file)
		fclose(trace_file);
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
			_STARPU_MSG("error while opening %s\n", options->out_paje_path);
			perror("fopen");
			exit(1);
		}

#ifdef STARPU_HAVE_POTI
#ifdef HAVE_POTI_INIT_CUSTOM
		fclose(out_paje_file);
		poti_init_custom (options->out_paje_path,
				  0, //if false, allow extended events
				  1, //if true, an old header (pj_dump -n)
				  0, //if false, the trace has no comments
				  1, //if true, events have aliases
				  1);//if true, relative timestamps
#else
		poti_init (out_paje_file);
#endif
#endif
		_starpu_fxt_write_paje_header(out_paje_file, options);
	}
	else
	{
		out_paje_file = NULL;
	}

	/* create lists for symbols (kernel states) and communications */
	_starpu_symbol_name_list_init(&symbol_list);
	_starpu_communication_list_init(&communication_list);
	if (!options->no_flops)
		_starpu_computation_list_init(&computation_list);
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
		STARPU_ABORT_MSG("Failed to open '%s' (err %s)", filename_in, strerror(errno));
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
	_starpu_fxt_data_file_init(options);
	_starpu_fxt_trace_file_init(options);

	_starpu_fxt_paje_file_init(options);

	if (options->ninputfiles == 0)
	{
		return;
	}
	else if (options->ninputfiles == 1)
	{
		/* we usually only have a single trace */
		uint64_t file_start_time = _starpu_fxt_find_start_time(options->filenames[0]);
		options->file_prefix = strdup("");
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
						_STARPU_MSG("Warning: traces are coming from different run so we will not try to display MPI communications.\n");
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

			free(options->file_prefix);
			options->file_prefix = strdup(file_prefix);
			options->file_offset = offsets[inputfile];
			options->file_rank = filerank;

			_starpu_fxt_parse_new_file(options->filenames[inputfile], options);
		}

		/* display the MPI transfers if possible */
		if (display_mpi)
			_starpu_fxt_display_mpi_transfers(options, rank_k, out_paje_file);
	}

	/* close the different files */
	_starpu_fxt_paje_file_close();
	_starpu_fxt_activity_file_close();
	_starpu_fxt_distrib_file_close(options);
	_starpu_fxt_anim_file_close();
	_starpu_fxt_tasks_file_close();
	_starpu_fxt_data_file_close();
	_starpu_fxt_trace_file_close();

	_starpu_fxt_dag_terminate();

	options->nworkers = nworkers;
	free(options->file_prefix);
}

#define DATA_STR_MAX_SIZE 15

struct parse_task
{
	unsigned exec_time;
	unsigned data_total;
	unsigned workerid;
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
		_STARPU_MALLOC(kernel, sizeof(*kernel));
		kernel->name = strdup(codelet_name);
		//fprintf(stderr, "%s\n", kernel->name);
		kernel->file = fopen(codelet_name, "w+");
		if(!kernel->file)
		{
			STARPU_ABORT_MSG("Failed to open '%s' (err %s)", codelet_name, strerror(errno));
		}
		HASH_ADD_STR(kernels, name, kernel);
		fprintf(codelet_list, "%s\n", codelet_name);
	}
	double time = pt.exec_time * NANO_SEC_TO_MILI_SEC;
	fprintf(kernel->file, "%lf %u %u\n", time, pt.data_total, pt.workerid);
}

void starpu_fxt_write_data_trace(char *filename_in)
{
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0)
	{
		STARPU_ABORT_MSG("Failed to open '%s' (err %s)", filename_in, strerror(errno));
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
		STARPU_ABORT_MSG("Failed to open '%s' (err %s)", "codelet_list", strerror(errno));
	}

	fxt_blockev_t block;
	block = fxt_blockev_enter(fut);

	struct fxt_ev_64 ev;

	int workerid=-1;
	unsigned long has_name = 0;

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
			register_worker_id(0 /* TODO: Add nodeid here instead */, ev.param[6], ev.param[1], ev.param[5]);
			break;

		case _STARPU_FUT_START_CODELET_BODY:
			workerid = ev.param[2];
			tasks[workerid].workerid = (unsigned)workerid;
			tasks[workerid].exec_time = ev.time;
			has_name = ev.param[4];
			tasks[workerid].codelet_name = strdup(has_name ? get_fxt_string(&ev, 5): "unknown");
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
			_STARPU_MSG("unknown event.. %x at time %llx WITH OFFSET %llx\n",
				    (unsigned)ev.code, (long long unsigned)ev.time, (long long unsigned)(ev.time));
#endif
			break;
		}
	}

#ifdef HAVE_FXT_CLOSE
	fxt_close(fut);
#endif
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

	struct starpu_data_trace_kernel *kernel=NULL, *tmp=NULL;
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
