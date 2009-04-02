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

#ifndef __DEQUE_MODELING_POLICY_H__
#define __DEQUE_MODELING_POLICY_H__

#include <core/workers.h>
#include <core/mechanisms/queues.h>
#include <core/mechanisms/fifo_queues.h>

void initialize_dm_policy(struct machine_config_s *config,
 __attribute__ ((unused)) struct sched_policy_s *_policy);

struct jobq_s *get_local_queue_dm(struct sched_policy_s *policy __attribute__ ((unused)));

#endif // __DEQUE_MODELING_POLICY_H__
