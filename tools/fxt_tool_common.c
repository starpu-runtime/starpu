/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

#include "fxt_tool.h"

static char *cpus_worker_colors[MAXWORKERS] = {"/greens9/7", "/greens9/6", "/greens9/5", "/greens9/4",  "/greens9/9", "/greens9/3",  "/greens9/2",  "/greens9/1"  };
static char *cuda_worker_colors[MAXWORKERS] = {"/ylorrd9/9", "/ylorrd9/6", "/ylorrd9/3", "/ylorrd9/1", "/ylorrd9/8", "/ylorrd9/7", "/ylorrd9/4", "/ylorrd9/2",  "/ylorrd9/1"};
static char *opencl_worker_colors[MAXWORKERS] = {"/blues9/9", "/blues9/6", "/blues9/3", "/blues9/1", "/blues9/8", "/blues9/7", "/blues9/4", "/blues9/2",  "/blues9/1"};
static char *other_worker_colors[MAXWORKERS] = {"/greys9/9", "/greys9/8", "/greys9/7", "/greys9/6"};
static char *worker_colors[MAXWORKERS];

static unsigned opencl_index = 0;
static unsigned cuda_index = 0;
static unsigned cpus_index = 0;
static unsigned other_index = 0;

void set_next_other_worker_color(int workerid)
{
	worker_colors[workerid] = other_worker_colors[other_index++];
}

void set_next_cpu_worker_color(int workerid)
{
	worker_colors[workerid] = cpus_worker_colors[cpus_index++];
}

void set_next_cuda_worker_color(int workerid)
{
	worker_colors[workerid] = cuda_worker_colors[cuda_index++];
}

void set_next_opencl_worker_color(int workerid)
{
	worker_colors[workerid] = opencl_worker_colors[opencl_index++];
}

const char *get_worker_color(int workerid)
{
	return worker_colors[workerid];
}

unsigned get_colour_symbol_red(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = _starpu_crc32_string(name, 0);
	return (unsigned)_starpu_crc32_string("red", hash_symbol) % 1024;
}

unsigned get_colour_symbol_green(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = _starpu_crc32_string(name, 0);
	return (unsigned)_starpu_crc32_string("green", hash_symbol) % 1024;
}

unsigned get_colour_symbol_blue(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = _starpu_crc32_string(name, 0);
	return (unsigned)_starpu_crc32_string("blue", hash_symbol) % 1024;
}



/* This must be called when we start handling a new trace */
void reinit_colors(void)
{
	other_index = 0;
	cpus_index = 0;
	cuda_index = 0;
}

uint64_t find_start_time(char *filename_in)
{
	/* Open the trace file */
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0) {
	        perror("open failed :");
	        exit(-1);
	}

	static fxt_t fut;
	fut = fxt_fdopen(fd_in);
	if (!fut) {
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


