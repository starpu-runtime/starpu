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

#include "fxt-tool.h"

static char *cpus_worker_colors[MAXWORKERS] = {"/greens9/7", "/greens9/6", "/greens9/5", "/greens9/4",  "/greens9/9", "/greens9/3",  "/greens9/2",  "/greens9/1"  };
static char *cuda_worker_colors[MAXWORKERS] = {"/ylorrd9/9", "/ylorrd9/6", "/ylorrd9/3", "/ylorrd9/1", "/ylorrd9/8", "/ylorrd9/7", "/ylorrd9/4", "/ylorrd9/2",  "/ylorrd9/1"};
static char *other_worker_colors[MAXWORKERS] = {"/greys9/9", "/greys9/8", "/greys9/7", "/greys9/6"};
static char *worker_colors[MAXWORKERS];

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

const char *get_worker_color(int workerid)
{
	return worker_colors[workerid];
}

unsigned get_colour_symbol_red(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = crc32_string(name, 0);
	return (unsigned)crc32_string("red", hash_symbol) % 1024;
}

unsigned get_colour_symbol_green(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = crc32_string(name, 0);
	return (unsigned)crc32_string("green", hash_symbol) % 1024;
}

unsigned get_colour_symbol_blue(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = crc32_string(name, 0);
	return (unsigned)crc32_string("blue", hash_symbol) % 1024;
}



/* This must be called when we start handling a new trace */
void reinit_colors(void)
{
	other_index = 0;
	cpus_index = 0;
	cuda_index = 0;
}
