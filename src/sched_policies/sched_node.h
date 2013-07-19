/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Simon Archipoff
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

#ifndef __SCHED_NODE_H__
#define __SCHED_NODE_H__

#include <starpu_sched_node.h>


/* lock and unlock drivers for modifying schedulers */
void _starpu_sched_node_lock_all_workers(void);
void _starpu_sched_node_unlock_all_workers(void);
void _starpu_sched_node_lock_worker(int workerid);
void _starpu_sched_node_unlock_worker(int workerid);


struct _starpu_worker * _starpu_sched_node_worker_get_worker(struct starpu_sched_node *);
struct _starpu_combined_worker * _starpu_sched_node_combined_worker_get_combined_worker(struct starpu_sched_node * worker_node);

struct starpu_bitmap * _starpu_get_worker_mask(unsigned sched_ctx_id);

#endif
