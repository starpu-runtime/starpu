/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
#include <core/mechanisms/deque_queues.h>

extern struct starpu_sched_policy_s _starpu_sched_dm_policy;

#endif // __DEQUE_MODELING_POLICY_H__
