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

#define _STARPU_MALLOC(p, s) do {p = malloc(s);} while (0)
#define _STARPU_CALLOC(p, n, s) do {p = calloc(n, s);} while (0)
#define _STARPU_MALLOC_CAST(p, s, t) do {p = (t) malloc(s);} while (0)

#ifndef NOCONFIG
#include <common/config.h>
#else
#define _GNU_SOURCE 1
// Assuming recent simgrid
#endif
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <common/list.h>
#include <common/prio_list.h>

#include <simgrid/modelchecker.h>

#include <xbt/base.h>

#include <simgrid/actor.h>
#include <simgrid/host.h>
#include <simgrid/engine.h>
#include <xbt/config.h>

#include <simgrid/mutex.h>
#include <simgrid/cond.h>

#include <common/rbtree.c>

#ifndef NLISTS
#define NLISTS 1
#endif
#ifndef NITERS
#define NITERS 1
#endif
#ifndef NTHREADS
#define NTHREADS 2
#endif
#ifndef NELEMENTS
#define NELEMENTS 4
#endif

// MC_ignore

sg_mutex_t mutex[NLISTS];

LIST_TYPE(foo,
		unsigned prio;
		unsigned back;	/* Push at back instead of front? */
	 );
PRIO_LIST_TYPE(foo, prio);

struct foo_prio_list mylist[NLISTS];

void check_list_prio(struct foo_prio_list *list)
{
	struct foo *cur;
	unsigned lastprio = UINT_MAX;
	unsigned back = 0;
	for (cur  = foo_prio_list_begin(list);
	     cur != foo_prio_list_end(list);
	     cur  = foo_prio_list_next(list, cur))
	{
		if (cur->prio == lastprio)
                        /* For same prio, back elements should never get before
                         * front elements */
			MC_assert(!(back && !cur->back));
		else
			MC_assert(lastprio > cur->prio);
		lastprio = cur->prio;
		back = cur->back;
	}
}

void worker(int argc, char *argv[])
{
	unsigned myrank = atoi(argv[0]);
	unsigned i, n, l, iter;
	struct foo *elem;
	struct drand48_data buffer;
	long res;

	srand48_r(myrank, &buffer);

	l = myrank%NLISTS;

	for (iter = 0; iter < NITERS; iter++)
	{
		for (i = 0; i < NELEMENTS; i++)
		{
			elem = malloc(sizeof(*elem));
			lrand48_r(&buffer, &res);
			elem->prio = res%10;
			lrand48_r(&buffer, &res);
			elem->back = res%2;
			sg_mutex_lock(mutex[l]);
			if (elem->back)
				foo_prio_list_push_back(&mylist[l], elem);
			else
				foo_prio_list_push_front(&mylist[l], elem);
			check_list_prio(&mylist[l]);
			sg_mutex_unlock(mutex[l]);
		}

		for (i = 0; i < NELEMENTS; i++)
		{
			lrand48_r(&buffer, &res);
			n = res%(NELEMENTS-i);

			sg_mutex_lock(mutex[l]);
			for (elem  = foo_prio_list_begin(&mylist[l]);
			     n--;
			     elem  = foo_prio_list_next(&mylist[l], elem))
				;
			foo_prio_list_erase(&mylist[l], elem);
			check_list_prio(&mylist[l]);
			sg_mutex_unlock(mutex[l]);
		}

		/* horrible way to wait for list getting empty */
		sg_actor_sleep_for(1000);
	}
}

void master(int argc, char *argv[])
{
}

int main(int argc, char *argv[])
{
	unsigned l, i;

	if (argc < 3)
	{
		fprintf(stderr,"usage: %s platform.xml host\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	srand48(0);
	simgrid_init(&argc, argv);
	sg_cfg_set_int("contexts/stack-size", 128);
	simgrid_load_platform(argv[1]);

	for (l = 0; l < NLISTS; l++)
	{
		mutex[l] = sg_mutex_init();
		foo_prio_list_init(&mylist[l]);
	}

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
