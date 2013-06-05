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

#ifndef __DRIVER_SCC_COMMON_H__
#define __DRIVER_SCC_COMMON_H__

#include <common/config.h>


#ifdef STARPU_USE_SCC

#include <RCCE_lib.h>

#include <drivers/mp_common/mp_common.h>

#define STARPU_TO_SCC_SINK_ID(id) (id) < RCCE_ue() ? (id) : ((id) + 1)

int _starpu_scc_common_mp_init();

void *_starpu_scc_common_get_shared_memory_addr();
void _starpu_scc_common_unmap_shared_memory();
int _starpu_scc_common_is_in_shared_memory(void *ptr);

int _starpu_scc_common_is_mp_initialized();

int _starpu_scc_common_get_src_node_id();
int _starpu_scc_common_is_src_node();

void _starpu_scc_common_send(const struct _starpu_mp_node *node, void *msg, int len);
void _starpu_scc_common_recv(const struct _starpu_mp_node *node, void *msg, int len);

void _starpu_scc_common_report_rcce_error(const char *func, const char *file, const int line, const int err_no);

#endif /* STARPU_USE_SCC */


#endif /* __DRIVER_SCC_COMMON_H__ */
