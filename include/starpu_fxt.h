/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2013,2016                           Inria
 * Copyright (C) 2013                                     Joris Pablo
 * Copyright (C) 2010-2015,2017,2018                      CNRS
 * Copyright (C) 2010-2011,2013-2018                      Université de Bordeaux
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

#include <starpu_perfmodel.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define STARPU_FXT_MAX_FILES	64

struct starpu_fxt_codelet_event
{
	char symbol[256];
	int workerid;
	char perfmodel_archname[256];
	uint32_t hash;
	size_t size;
	float time;
};

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
	char *dag_path;
	char *tasks_path;
	char *data_path;
	char *anim_path;
	char *states_path;

	char *file_prefix;
	uint64_t file_offset;
	int file_rank;

	char worker_names[STARPU_NMAXWORKERS][256];
	struct starpu_perfmodel_arch worker_archtypes[STARPU_NMAXWORKERS];
	int nworkers;

	struct starpu_fxt_codelet_event **dumped_codelets;
	long dumped_codelets_count;
};

void starpu_fxt_options_init(struct starpu_fxt_options *options);
void starpu_fxt_generate_trace(struct starpu_fxt_options *options);
void starpu_fxt_autostart_profiling(int autostart);
void starpu_fxt_start_profiling(void);
void starpu_fxt_stop_profiling(void);
void starpu_fxt_write_data_trace(char *filename_in);
void starpu_fxt_trace_user_event(unsigned long code);
void starpu_fxt_trace_user_event_string(const char *s);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_FXT_H__ */
