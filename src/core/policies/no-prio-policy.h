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

#ifndef __NO_PRIO_POLICY_H__
#define __NO_PRIO_POLICY_H__

#include <core/workers.h>
#include <core/mechanisms/fifo_queues.h>

void initialize_no_prio_policy(struct machine_config_s *config, struct sched_policy_s *policy);
//void set_local_queue_eager(struct jobq_s *jobq);
struct jobq_s *get_local_queue_no_prio(struct sched_policy_s *policy);

#endif // __NO_PRIO_POLICY_H__
