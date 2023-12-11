/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018-2020  Federal University of Rio Grande do Sul (UFRGS)
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

#ifndef __STARPU__FXT_H__
#define __STARPU__FXT_H__

/** @file */

#include <starpu.h>
#include <starpu_config.h>
#include <common/config.h>

#ifdef STARPU_USE_FXT

#include <search.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <common/uthash.h>
#include <common/fxt.h>
#include <common/list.h>
#include "../mpi/src/starpu_mpi_fxt.h"
#include <starpu.h>
#include "../../../include/starpu_fxt.h"

#ifdef STARPU_HAVE_POTI
#include <poti.h>
#define STARPU_POTI_STR_LEN 200
#endif
#define STARPU_TRACE_STR_LEN 200

#pragma GCC visibility push(hidden)

#define MAX_PARAMETERS 8
struct _thread_info
{
	UT_hash_handle hh;
	long unsigned int tid;
	int worker;
	char symbol[(FXT_MAX_PARAMS-5)*sizeof(unsigned long)];
	int codelet_parameter;
	char codelet_parameter_description[MAX_PARAMETERS][FXT_MAX_PARAMS*sizeof(unsigned long)];
	/* Start time of last codelet for this worker */
	double last_codelet_start;
	/* End time of last codelet for this worker */
	double last_codelet_end;
	struct _starpu_computation *ongoing_computation;
	double accumulated_sleep_time;
	double accumulated_exec_time;
	double last_activity_flush_timestamp;

	// data used by starpu_fxt_write_data_trace_in_dir()
	unsigned exec_time;
	unsigned data_total;
	unsigned workerid;
	char *codelet_name;
};
extern struct _thread_info *_thread_infos;

extern char _starpu_last_codelet_symbol[STARPU_NMAXWORKERS][(FXT_MAX_PARAMS-5)*sizeof(unsigned long)];

void _starpu_fxt_dag_init(char *dag_filename);
void _starpu_fxt_dag_terminate(void);
void _starpu_fxt_dag_add_tag(const char *prefix, uint64_t tag, unsigned long job_id, const char *label);
void _starpu_fxt_dag_add_tag_deps(const char *prefix, uint64_t child, uint64_t father, const char *label);
void _starpu_fxt_dag_set_tag_done(const char *prefix, uint64_t tag, const char *color, const char *fontcolor);
void _starpu_fxt_dag_add_task_deps(const char *prefix, unsigned long dep_prev, unsigned long dep_succ, const char *label);
void _starpu_fxt_dag_add_task_end_dep(const char *prefix, unsigned long prev, unsigned long succ);
void _starpu_fxt_dag_set_task_name(const char *prefix, unsigned long job_id, const char *label, const char *color, const char *fontcolor);
#ifdef STARPU_BUBBLE
void _starpu_fxt_dag_set_task_bubble(const char *prefix, unsigned long job_id, int is_bubble, unsigned long bubble_parent);
#endif
void _starpu_fxt_dag_set_task_line(const char *prefix, unsigned long job_id, const char *file, int line);
void _starpu_fxt_dag_add_send(int src, unsigned long dep_prev, unsigned long tag, unsigned long id);
void _starpu_fxt_dag_add_receive(int dst, unsigned long dep_prev, unsigned long tag, unsigned long id);
void _starpu_fxt_dag_add_sync_point(void);
unsigned _starpu_fxt_data_get_coord(unsigned long handle, int mpi_rank, unsigned dim);
const char * _starpu_fxt_data_get_name(unsigned long handle, int mpi_rank);

void _starpu_convert_numa_nodes_bitmap_to_str(long bitmap, char str[]);

/*
 *	MPI
 */

struct starpu_fxt_mpi_offset _starpu_fxt_mpi_find_sync_points(char *filename_in, int *key, int *rank);
void _starpu_fxt_mpi_add_send_transfer(int src, int dst, long mpi_tag, size_t size, float date, long jobid, unsigned long handle, unsigned type, int prio);
void _starpu_fxt_mpi_send_transfer_set_numa_node(int src, int dest, long jobid, long numa_nodes_bitmap);
void _starpu_fxt_mpi_add_recv_transfer(int src, int dst, long mpi_tag, float date, long jobid, unsigned long handle);
void _starpu_fxt_mpi_recv_transfer_set_numa_node(int src, int dst, long jobid, long numa_nodes_bitmap);
void _starpu_fxt_display_mpi_transfers(struct starpu_fxt_options *options, int *ranks, FILE *out_paje_file, FILE* out_comms_file);

void _starpu_fxt_write_paje_header(FILE *file, struct starpu_fxt_options *options);

extern int _starpu_poti_extendedSetState;
extern int _starpu_poti_semiExtendedSetState;
extern int _starpu_poti_MemoryEvent;
extern int _starpu_poti_CommLinkStart;
extern int _starpu_poti_MpiLinkStart;
extern int _starpu_poti_checkPointState;
extern int _starpu_poti_JobState;

/*
 * Animation
 */
void _starpu_fxt_component_print_header(FILE *output);
void _starpu_fxt_component_new(uint64_t component, char *name);
void _starpu_fxt_component_connect(uint64_t parent, uint64_t child);
void _starpu_fxt_component_update_ntasks(unsigned nsubmitted, unsigned curq_size);
void _starpu_fxt_component_push(FILE *output, struct starpu_fxt_options *options, double timestamp, long unsigned int tid, int workerid, uint64_t from, uint64_t to, uint64_t task, unsigned prio);
void _starpu_fxt_component_pull(FILE *output, struct starpu_fxt_options *options, double timestamp, long unsigned int tid, int workerid, uint64_t from, uint64_t to, uint64_t task, unsigned prio);
void _starpu_fxt_component_dump(FILE *output);
void _starpu_fxt_component_finish(FILE *output);
void _starpu_fxt_component_deinit(void);

#pragma GCC visibility pop

#endif // STARPU_USE_FXT

#endif // __STARPU__FXT_H__
