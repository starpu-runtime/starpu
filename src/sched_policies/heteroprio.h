/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __SCHED_HETEROPRIO_H__
#define __SCHED_HETEROPRIO_H__

#include <schedulers/starpu_heteroprio.h>

#define CODELET_MAX_NAME_LENGTH 32
#define HETEROPRIO_MAX_PRIO 100
#define LAHETEROPRIO_MAX_WORKER_GROUPS 10

#define AUTOHETEROPRIO_NO_NAME "NO_NAME"

// will tend to ignore tasks older than this when measuring values such as NOD, execution time, etc.
// i.e. if there are more than STARPU_AUTOHETEROPRIO_RELEVANT_TASK_LIFE of the same type
#define AUTOHETEROPRIO_RELEVANT_TASK_LIFE 256

#define AUTOHETEROPRIO_RELEVANT_SAMPLE_SIZE 16

#define AUTOHETEROPRIO_EXTREMELY_LONG_TIME 999999999999999.0
#define AUTOHETEROPRIO_LONG_TIME 100000000.0
#define AUTOHETEROPRIO_FAIR_TIME 1000.0

#define AUTOHETEROPRIO_DEFAULT_TASK_TIME AUTOHETEROPRIO_FAIR_TIME

// at the end of the execution, if the sum of all worker profiling times is superior to this, the times will be compressed so that no time exceeds this one
// (probably in us)
#define AUTOHETEROPRIO_MAX_WORKER_PROFILING_TIME 1000000000.0

#endif // __SCHED_HETEROPRIO_H__
