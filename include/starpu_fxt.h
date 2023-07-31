/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Joris Pablo
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2020       Federal University of Rio Grande do Sul (UFRGS)
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

#ifndef __STARPU_FXT_H__
#define __STARPU_FXT_H__

#include <starpu_config.h>
#include <starpu_perfmodel.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_FxT_Support FxT Support
   @{
*/

/**
   todo
*/
struct starpu_fxt_codelet_event
{
	char symbol[2048];
	int workerid;
	char perfmodel_archname[256];
	uint32_t hash;
	size_t size;
	float time;
};

/**
   Store information related to clock synchronizations: mainly the offset to apply to each time.
*/
struct starpu_fxt_mpi_offset
{
	uint64_t local_time_start; /**< node time for the barrier at the beginning of the program */
	int64_t offset_start;	   /**< offset to apply to node time, computed at the beginning of the program */
	uint64_t local_time_end;   /**< node time for the barrier at the end of the program (optional) */
	int64_t offset_end;	   /**< offset to apply to node time, computed at the end of the program (optional) */
	int nb_barriers;	   /**< number of barriers to synchronize clocks during the execution of the program
			   (can be 0, 1 or 2) */
};

/**
   todo
*/
struct starpu_fxt_options
{
	unsigned per_task_colour;
	unsigned no_events;
	unsigned no_counter;
	unsigned no_bus;
	unsigned no_flops;
	unsigned ninputfiles;
	unsigned no_smooth;
	unsigned no_acquire;
	unsigned memory_states;
	unsigned internal;
	unsigned label_deps;
	char *filenames[STARPU_FXT_MAX_FILES];
	char *out_paje_path;
	char *distrib_time_path;
	char *activity_path;
	char *sched_tasks_path;
	char *dag_path;
	char *tasks_path;
	char *data_path;
	char *papi_path;
	char *comms_path;
	char *number_events_path;
	char *anim_path;
	char *states_path;
	char *dir;
	char worker_names[STARPU_NMAXWORKERS][256];
	int nworkers;
	struct starpu_perfmodel_arch worker_archtypes[STARPU_NMAXWORKERS];

	/**
	   In case we are going to gather multiple traces (e.g in the case of
	   MPI processes), we may need to prefix the name of the containers.
	*/
	char *file_prefix;

	/**
	   In case we are going to gather multiple traces (e.g in the case of
	   MPI processes), we may need to synchronize clocks and apply an offset.
	*/
	struct starpu_fxt_mpi_offset file_offset;

	/**
	   In case we are going to gather multiple traces (e.g in the case of
	   MPI processes), this variable stores the MPI rank of the trace file.
	*/
	int file_rank;

	/**
	   In case we want to dump the list of codelets to an external tool
	*/
	struct starpu_fxt_codelet_event **dumped_codelets;

	/**
	   In case we want to dump the list of codelets to an external tool, number
	   of dumped codelets.
	*/
	long dumped_codelets_count;
};

void starpu_fxt_options_init(struct starpu_fxt_options *options);
void starpu_fxt_options_shutdown(struct starpu_fxt_options *options);
void starpu_fxt_generate_trace(struct starpu_fxt_options *options);

/**
   Determine whether profiling should be started by starpu_init(), or only when
   starpu_fxt_start_profiling() is called. \p autostart should be 1 to do so, or 0 to
   prevent it.
   This function has to be called before starpu_init().
   See \ref LimitingScopeTrace for more details.
*/
void starpu_fxt_autostart_profiling(int autostart);

/**
   Start recording the trace. The trace is by default started from
   starpu_init() call, but can be paused by using
   starpu_fxt_stop_profiling(), in which case
   starpu_fxt_start_profiling() should be called to resume recording
   events.
   See \ref LimitingScopeTrace for more details.
*/
void starpu_fxt_start_profiling(void);

/**
   Stop recording the trace. The trace is by default stopped when calling
   starpu_shutdown(). starpu_fxt_stop_profiling() can however be used to
   stop it earlier. starpu_fxt_start_profiling() can then be called to
   start recording it again, etc.
   See \ref LimitingScopeTrace for more details.
*/
void starpu_fxt_stop_profiling(void);

void starpu_fxt_write_data_trace(char *filename_in);
void starpu_fxt_write_data_trace_in_dir(char *filename_in, char *dir);

/**
    Wrapper to get value of env variable STARPU_FXT_TRACE
*/
int starpu_fxt_is_enabled(void);

/**
   Add an event in the execution trace if FxT is enabled.
   See \ref CreatingAGanttDiagram for more details.
*/
void starpu_fxt_trace_user_event(unsigned long code);

/**
   Add a string event in the execution trace if FxT is enabled.
   See \ref CreatingAGanttDiagram for more details.
*/
void starpu_fxt_trace_user_event_string(const char *s);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_FXT_H__ */
