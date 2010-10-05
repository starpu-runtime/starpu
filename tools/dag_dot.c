/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#include <stdio.h>
#include <stdint.h>

#include "fxt_tool.h"

static char *out_path = "dag.dot";
static FILE *out_file;
static unsigned cluster_cnt;

void init_dag_dot(void)
{
	/* create a new file */
	out_file = fopen(out_path, "w+");
	cluster_cnt = 0;

	fprintf(out_file, "digraph G {\n");
	fprintf(out_file, "\tcolor=white\n");
	fprintf(out_file, "\trankdir=LR;\n");

	/* Create a new cluster */
	fprintf(out_file, "subgraph cluster_%d {\n", cluster_cnt);
	fprintf(out_file, "\tcolor=black;\n");
}

void terminate_dat_dot(void)
{
	/* Close the last cluster */
	fprintf(out_file, "}\n");
	/* Close the graph */
	fprintf(out_file, "}\n");
	fclose(out_file);
}

void add_deps(uint64_t child, uint64_t father)
{
	fprintf(out_file, "\t \"tag_%llx\"->\"tag_%llx\"\n", 
		(unsigned long long)father, (unsigned long long)child);
}

void add_task_deps(unsigned long dep_prev, unsigned long dep_succ)
{
	fprintf(out_file, "\t \"task_%lx\"->\"task_%lx\"\n", dep_prev, dep_succ);
} 

void dot_set_tag_done(uint64_t tag, const char *color)
{

	fprintf(out_file, "\t \"tag_%llx\" \[ style=filled, label=\"\", color=\"%s\"]\n", 
		(unsigned long long)tag, color);
}

void dot_set_task_done(unsigned long job_id, const char *label, const char *color)
{
	fprintf(out_file, "\t \"task_%lx\" \[ style=filled, label=\"%s\", color=\"%s\"]\n", job_id, label, color);
}

void dot_add_sync_point(void)
{
	/* Close the previous cluster */
	fprintf(out_file, "}\n");

	cluster_cnt++;

	/* Create a new cluster */
	fprintf(out_file, "subgraph cluster_%d {\n", cluster_cnt);
	fprintf(out_file, "\tcolor=black;\n");
}
