/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <stdint.h>
#include <common/config.h>

#ifdef STARPU_USE_FXT

#include "starpu_fxt.h"

static FILE *out_file;
static unsigned cluster_cnt;

void _starpu_fxt_dag_init(char *out_path)
{
	if (!out_path)
	{
		out_file = NULL;
		return;
	}

	/* create a new file */
	out_file = fopen(out_path, "w+");
	if (!out_file)
	{
		_STARPU_MSG("error while opening %s\n", out_path);
		perror("fopen");
		exit(1);
	}
	cluster_cnt = 0;

	fprintf(out_file, "digraph G {\n");
	fprintf(out_file, "\tcolor=white\n");
	fprintf(out_file, "\trankdir=LR;\n");

	/* Create a new cluster */
	fprintf(out_file, "subgraph cluster_%u {\n", cluster_cnt);
	fprintf(out_file, "\tcolor=black;\n");
}

void _starpu_fxt_dag_terminate(void)
{
	if (!out_file)
		return;

	/* Close the last cluster */
	fprintf(out_file, "}\n");
	/* Close the graph */
	fprintf(out_file, "}\n");
	fclose(out_file);
}

void _starpu_fxt_dag_add_tag(const char *prefix, uint64_t tag, unsigned long job_id, const char *label)
{
	if (out_file)
	{
		if (label)
			fprintf(out_file, "\t \"tag_%s%llx\"->\"task_%s%lu\"->\"tag_%s%llx\" [style=dashed] [label=\"%s\"]\n", prefix, (unsigned long long)tag, prefix, (unsigned long)job_id, prefix, (unsigned long long) tag, label);
		else
			fprintf(out_file, "\t \"tag_%s%llx\"->\"task_%s%lu\"->\"tag_%s%llx\" [style=dashed]\n", prefix, (unsigned long long)tag, prefix, (unsigned long)job_id, prefix, (unsigned long long) tag);
	}
}

void _starpu_fxt_dag_add_tag_deps(const char *prefix, uint64_t child, uint64_t father, const char *label)
{
	if (out_file)
	{
		if (label)
			fprintf(out_file, "\t \"tag_%s%llx\"->\"tag_%s%llx\" [label=\"%s\"]\n", prefix, (unsigned long long)father, prefix, (unsigned long long)child, label);
		else
			fprintf(out_file, "\t \"tag_%s%llx\"->\"tag_%s%llx\"\n", prefix, (unsigned long long)father, prefix, (unsigned long long)child);
	}
}

void _starpu_fxt_dag_add_task_deps(const char *prefix, unsigned long dep_prev, unsigned long dep_succ, const char *label)
{
	if (out_file)
	{
		if (label)
			fprintf(out_file, "\t \"task_%s%lu\"->\"task_%s%lu\" [label=\"%s\"]\n", prefix, dep_prev, prefix, dep_succ, label);
		else
			fprintf(out_file, "\t \"task_%s%lu\"->\"task_%s%lu\"\n", prefix, dep_prev, prefix, dep_succ);
	}
}

void _starpu_fxt_dag_set_tag_done(const char *prefix, uint64_t tag, const char *color)
{
	if (out_file)
		fprintf(out_file, "\t \"tag_%s%llx\" [ style=filled, fillcolor=\"%s\"]\n",
			prefix, (unsigned long long)tag, color);
}

void _starpu_fxt_dag_set_task_name(const char *prefix, unsigned long job_id, const char *label, const char *color)
{
	if (out_file)
		fprintf(out_file, "\t \"task_%s%lu\" [ style=filled, label=\"%s\", fillcolor=\"%s\"]\n", prefix, job_id, label, color);
}

void _starpu_fxt_dag_add_send(int src, unsigned long dep_prev, unsigned long tag, unsigned long id)
{
	if (out_file)
		fprintf(out_file, "\t \"task_%d_%lu\"->\"mpi_%lu_%lu\"\n", src, dep_prev, tag, id);
}

void _starpu_fxt_dag_add_receive(int dst, unsigned long dep_prev, unsigned long tag, unsigned long id)
{
	if (out_file)
		fprintf(out_file, "\t \"mpi_%lu_%lu\"->\"task_%d_%lu\"\n", tag, id, dst, dep_prev);
}

void _starpu_fxt_dag_add_sync_point(void)
{
	if (!out_file)
		return;

	/* Close the previous cluster */
	fprintf(out_file, "}\n");

	cluster_cnt++;

	/* Create a new cluster */
	fprintf(out_file, "subgraph cluster_%u {\n", cluster_cnt);
	fprintf(out_file, "\tcolor=black;\n");
}

#endif /* STARPU_USE_FXT */
