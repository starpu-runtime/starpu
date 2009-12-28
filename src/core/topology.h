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

#ifndef __TOPOLOGY_H__
#define __TOPOLOGY_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <common/config.h>
#include <common/list.h>
#include <common/fxt.h>

#include <starpu.h>

/* TODO actually move this struct into this header */
struct machine_config_s;

/* This structure is "inspired" by the hwloc project
 * (see http://www.open-mpi.org/projects/hwloc/) */

struct starpu_topo_obj_t {
	/* global position */
	unsigned level;
	unsigned number;

	/* father */
	struct starpu_topo_obj_t *father;
	unsigned index;
	
	/* children */
	unsigned arity;
	struct starpu_topo_obj **children;
	struct starpu_topo_obj *first_child;
	struct starpu_topo_obj *last_child;

	/* cousins */
	struct topo_obj *next_cousin;
	struct topo_obj *prev_cousin;

	/* for the convenience of the scheduler */
	void *sched_data;

	/* flags */
	unsigned is_a_worker;
	struct worker_s *worker; /* (ignored if !is_a_worker) */
};

int starpu_build_topology(struct machine_config_s *config);

void starpu_destroy_topology(struct machine_config_s *config);

/* returns the number of physical cores */
unsigned topology_get_nhwcore(struct machine_config_s *config);

void bind_thread_on_cpu(struct machine_config_s *config, unsigned coreid);

#endif // __TOPOLOGY_H__
