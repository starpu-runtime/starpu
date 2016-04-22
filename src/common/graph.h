/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016  Université de Bordeaux
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

void _starpu_graph_init(void);
extern int _starpu_graph_record;

/* Add a job to the graph, called before any _starpu_graph_add_job_dep call */
void _starpu_graph_add_job(struct _starpu_job *job);

/* Add a dependency between jobs */
void _starpu_graph_add_job_dep(struct _starpu_job *job, struct _starpu_job *prev_job);

/* Remove a job from the graph */
void _starpu_graph_drop_job(struct _starpu_job *job);

/* Compute the depth of jobs in the graph */
/* This does not take job duration into account, just the number */
void _starpu_graph_compute_depths(void);

/* Apply func on each job of the graph */
void _starpu_graph_foreach(void (*func)(void *data, struct _starpu_job *job), void *data);
