/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_TCPIP_SOURCE_H__
#define __DRIVER_TCPIP_SOURCE_H__

/** @file */

#include <core/workers.h>
#include <drivers/mp_common/mp_common.h>
#include <datawizard/node_ops.h>

#pragma GCC visibility push(hidden)

void _starpu_tcpip_ms_preinit(void);

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
extern struct _starpu_node_ops _starpu_driver_tcpip_ms_node_ops;

/** Array of structures containing all the information useful to send
 * and receive information with devices */
struct _starpu_mp_node *_starpu_tcpip_ms_src_get_actual_thread_mp_node();

unsigned _starpu_tcpip_src_get_device_count();
void _starpu_init_tcpip_config(struct _starpu_machine_topology * topology, struct _starpu_machine_config *config, struct starpu_conf *user_conf, int no_mp_config);
void _starpu_tcpip_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);
void _starpu_tcpip_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);
void _starpu_deinit_tcpip_config(struct _starpu_machine_config *config);
void *_starpu_tcpip_src_worker(void *arg);

void _starpu_tcpip_source_init(struct _starpu_mp_node *node);
void _starpu_tcpip_source_deinit(struct _starpu_mp_node *node);

int _starpu_tcpip_is_direct_access_supported(unsigned node, unsigned handling_node);

#endif /* STARPU_USE_TCPIP_MASTER_SLAVE */

#pragma GCC visibility pop

#endif	/* __DRIVER_TCPIP_SOURCE_H__ */
