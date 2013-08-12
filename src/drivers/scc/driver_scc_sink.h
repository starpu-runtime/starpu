/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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

#ifndef __DRIVER_SCC_SINK_H__
#define __DRIVER_SCC_SINK_H__

#include <common/config.h>


#ifdef STARPU_USE_SCC

#include <drivers/mp_common/mp_common.h>

void _starpu_scc_sink_init(struct _starpu_mp_node *node);
void _starpu_scc_sink_launch_workers(struct _starpu_mp_node *node);
void _starpu_scc_sink_deinit(struct _starpu_mp_node *node);

void _starpu_scc_sink_send_to_device(const struct _starpu_mp_node *node, int dst_devid, void *msg, int len);
void _starpu_scc_sink_recv_from_device(const struct _starpu_mp_node *node, int src_devid, void *msg, int len);

void _starpu_scc_sink_bind_thread(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, cpu_set_t * cpuset, int coreid, pthread_t *thread);

void _starpu_scc_sink_execute(const struct _starpu_mp_node *node, void *arg, int arg_size);

void (*_starpu_scc_sink_lookup (const struct _starpu_mp_node * node STARPU_ATTRIBUTE_UNUSED, char* func_name))(void);

#endif /* STARPU_USE_SCC */


#endif /* __DRIVER_SCC_SINK_H__ */
