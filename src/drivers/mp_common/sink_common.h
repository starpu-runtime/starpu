/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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


#ifndef __SINK_COMMON_H__
#define __SINK_COMMON_H__

/** @file */

#include <common/config.h>

#ifdef STARPU_USE_MP

#include <drivers/mp_common/mp_common.h>

#pragma GCC visibility push(hidden)

/** Represent the topology of sink devices, contains useful information about
 * their capabilities
 * XXX: unused.
 */
struct _starpu_sink_topology
{
	unsigned nb_cpus;
};

struct arg_sink_thread
{
	struct _starpu_mp_node *node;
	int coreid;
};

void _starpu_sink_common_worker(void);

void (*_starpu_sink_common_cpu_lookup (const struct _starpu_mp_node * node STARPU_ATTRIBUTE_UNUSED, char* func_name))(void);
void _starpu_sink_common_execute(struct _starpu_mp_node *node, void *arg, int arg_size);

void _starpu_sink_common_allocate(const struct _starpu_mp_node *mp_node, void *arg, int arg_size);
void _starpu_sink_common_free(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, void *arg, int arg_size);

void _starpu_sink_common_map(const struct _starpu_mp_node *mp_node, void *arg, int arg_size);
void _starpu_sink_common_unmap(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, void *arg, int arg_size);

void* _starpu_sink_thread(void * thread_arg);

#pragma GCC visibility pop

#endif /* STARPU_USE_MP */

#endif /* __SINK_COMMON_H__ */
