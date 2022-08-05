/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define __COMMON_UTILS_H__
#define _STARPU_MALLOC(p, s) do {p = malloc(s);} while (0)
#define _STARPU_CALLOC(p, n, s) do {p = calloc(n, s);} while (0)
#define _STARPU_REALLOC(p, s) do {p = realloc(p, s);} while (0)
#define STARPU_HG_DISABLE_CHECKING(v) ((void) 0)
#define STARPU_HG_ENABLE_CHECKING(v) ((void) 0)
#define ANNOTATE_HAPPENS_AFTER(v) ((void) 0)
#define ANNOTATE_HAPPENS_BEFORE(v) ((void) 0)

#define STARPU_DEBUG_PREFIX "[starpu]"
#ifdef STARPU_VERBOSE
#  define _STARPU_DEBUG(fmt, ...) do { if (!_starpu_silent) {fprintf(stderr, STARPU_DEBUG_PREFIX"[%s] " fmt ,__starpu_func__ ,## __VA_ARGS__); fflush(stderr); }} while(0)
#else
#  define _STARPU_DEBUG(fmt, ...) do { } while (0)
#endif

#define STARPU_UYIELD() ((void)0)

#ifndef NOCONFIG
#include <common/config.h>
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
// Assuming recent simgrid
#endif
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <common/barrier.h>

#include <simgrid/modelchecker.h>

#include <xbt/base.h>
#include <simgrid/actor.h>
#include <simgrid/host.h>
#include <simgrid/engine.h>
#include <xbt/config.h>

#include <simgrid/mutex.h>
#include <simgrid/cond.h>

/* common/thread.c references these, but doesn't need to have them working anyway */
void _starpu_simgrid_thread_start(int argc, char *argv[])
{
}

size_t _starpu_default_stack_size = 8192;

void
_starpu_simgrid_set_stack_size(size_t stack_size)
{
}

starpu_sg_host_t _starpu_simgrid_get_host_by_name(const char *name)
{
	return NULL;
}

static void _starpu_clock_gettime(struct timespec *ts)
{
	double now = simgrid_get_clock();
	ts->tv_sec = floor(now);
	ts->tv_nsec = floor((now - ts->tv_sec) * 1000000000);
}

void starpu_sleep(float nb_sec)
{
	sg_actor_sleep_for(nb_sec);
}

#include <common/barrier.c>
#undef STARPU_DEBUG
int starpu_worker_get_id(void) { return 0; }
static inline unsigned _starpu_worker_mutex_is_sched_mutex(int workerid, starpu_pthread_mutex_t *mutex) { return 0; }
#include <common/thread.c>

#ifndef NTHREADS
#define NTHREADS 2
#endif

#ifndef NITERS
#define NITERS 1
#endif

struct _starpu_barrier barrier;

void worker(int argc, char *argv[])
{
	unsigned iter;

	for (iter = 0; iter < NITERS; iter++)
	{
		MC_assert(barrier.count <= NTHREADS);
		_starpu_barrier_wait(&barrier);
	}
}

#undef main
int main(int argc, char *argv[])
{
	unsigned i;

	if (argc < 3)
	{
		fprintf(stderr,"usage: %s platform.xml host\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	srand48(0);
	simgrid_init(&argc, argv);
	sg_cfg_set_int("contexts/stack-size", 128);
	simgrid_load_platform(argv[1]);

	_starpu_barrier_init(&barrier, NTHREADS);

	for (i = 0; i < NTHREADS; i++)
	{
		char *s;
		asprintf(&s, "%d\n", i);
		char **args = malloc(sizeof(char*)*2);
		args[0] = s;
		args[1] = NULL;
		sg_actor_create("test", sg_host_by_name(argv[2]), worker, 1, args);
	}

	simgrid_run();
	return 0;
}
