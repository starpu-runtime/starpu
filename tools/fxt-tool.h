/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __FXT_TOOL_H__
#define __FXT_TOOL_H__

#include <search.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <common/fxt.h>
#include <common/list.h>

#include "histo-paje.h"

#define MAXWORKERS      32
#define FACTOR  100

typedef enum {
	WORKING,
	FETCHING,
	PUSHING,
	IDLE
} worker_mode;

LIST_TYPE(event,
        uint64_t time;
	worker_mode mode;
);

LIST_TYPE(workq,
	uint64_t time;
	int diff;
	int current_size;
);

extern void init_dag_dot(void);
extern void terminate_dat_dot(void);
extern void add_deps(uint64_t child, uint64_t father);
extern void dot_set_tag_done(uint64_t tag, char *color);

extern void svg_engine_generate_output(event_list_t *events, 
	workq_list_t taskq, char **worker_name, unsigned nworkers,
	unsigned maxq_size, uint64_t _start_time, uint64_t _end_time,
	char *path);

#endif // __FXT_TOOL_H__
