/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#define STARPU_HAVE_SIMGRID_MSG_H
#define STARPU_HAVE_SIMGRID_SEMAPHORE_H
#define STARPU_HAVE_SIMGRID_MUTEX_H
#define STARPU_HAVE_SIMGRID_COND_H
#define STARPU_HAVE_SIMGRID_BARRIER_H
#define STARPU_HAVE_XBT_SYNCHRO_H
#define HAVE_SIMGRID_GET_CLOCK
#define HAVE_SG_ACTOR_SLEEP_FOR
#define HAVE_SG_CFG_SET_INT
#endif
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <common/list.h>
#include <common/prio_list.h>
#ifdef STARPU_HAVE_SIMGRID_MSG_H
#include <simgrid/msg.h>
#else
#include <msg/msg.h>
#endif
#include <simgrid/modelchecker.h>
#ifdef STARPU_HAVE_XBT_SYNCHRO_H
#include <xbt/synchro.h>
#else
#include <xbt/synchro_core.h>
#endif

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

#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
sg_mutex_t mutex[NLISTS];
#define mutex_lock(l) sg_mutex_lock(l)
#define mutex_unlock(l) sg_mutex_unlock(l)
#else
xbt_mutex_t mutex[NLISTS];
#define mutex_lock(l) xbt_mutex_acquire(l)
#define mutex_unlock(l) xbt_mutex_release(l)
#endif


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

int worker(int argc, char *argv[])
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
			mutex_lock(mutex[l]);
			if (elem->back)
				foo_prio_list_push_back(&mylist[l], elem);
			else
				foo_prio_list_push_front(&mylist[l], elem);
			check_list_prio(&mylist[l]);
			mutex_unlock(mutex[l]);
		}

		for (i = 0; i < NELEMENTS; i++)
		{
			lrand48_r(&buffer, &res);
			n = res%(NELEMENTS-i);

			mutex_lock(mutex[l]);
			for (elem  = foo_prio_list_begin(&mylist[l]);
			     n--;
			     elem  = foo_prio_list_next(&mylist[l], elem))
				;
			foo_prio_list_erase(&mylist[l], elem);
			check_list_prio(&mylist[l]);
			mutex_unlock(mutex[l]);
		}

		/* horrible way to wait for list getting empty */
#ifdef HAVE_SG_ACTOR_SLEEP_FOR
		sg_actor_sleep_for(1000);
#else
		MSG_process_sleep(1000);
#endif
	}

	return 0;
}

int master(int argc, char *argv[])
{
	unsigned i, l;

	for (l = 0; l < NLISTS; l++)
	{
#ifdef STARPU_HAVE_SIMGRID_MUTEX_H
		mutex[l] = sg_mutex_init();
#else
		mutex[l] = xbt_mutex_init();
#endif
		foo_prio_list_init(&mylist[l]);
	}

	for (i = 0; i < NTHREADS; i++)
	{
		char *s;
		asprintf(&s, "%d\n", i);
		char **args = malloc(sizeof(char*)*2);
		args[0] = s;
		args[1] = NULL;
		MSG_process_create_with_arguments("test", worker, NULL, MSG_host_self(), 1, args);
	}

	return 0;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		fprintf(stderr,"usage: %s platform.xml host\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	srand48(0);
	MSG_init(&argc, argv);
#ifdef HAVE_SG_CFG_SET_INT
	sg_cfg_set_int("contexts/stack-size", 128);
#elif SIMGRID_VERSION_MAJOR < 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR < 13)
	extern xbt_cfg_t _sg_cfg_set;
	xbt_cfg_set_int(_sg_cfg_set, "contexts/stack-size", 128);
#else
	xbt_cfg_set_int("contexts/stack-size", 128);
#endif
	MSG_create_environment(argv[1]);
	MSG_process_create("master", master, NULL, MSG_get_host_by_name(argv[2]));
	MSG_main();
	return 0;
}
