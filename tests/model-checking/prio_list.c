/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017  Université de Bordeaux
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
#define STARPU_ATTRIBUTE_UNUSED __attribute((__unused__))

#define _GNU_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <common/list.h>
#include <common/prio_list.h>
#include <simgrid/msg.h>
#include <simgrid/modelchecker.h>
#include <xbt/synchro.h>

#define N 2 /* number of threads */
#define M 4 /* number of elements */

// MC_ignore

xbt_mutex_t mutex;


LIST_TYPE(foo,
		unsigned prio;
		unsigned back;	/* Push at back instead of front? */
	 );
PRIO_LIST_TYPE(foo, prio);

struct foo_prio_list mylist;

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
	unsigned i, n;
	struct foo *elem;
	struct drand48_data buffer;
	long res;

	srand48_r(atoi(argv[0]), &buffer);

	for (i = 0; i < M; i++)
	{
		elem = malloc(sizeof(*elem));
		lrand48_r(&buffer, &res);
		elem->prio = res%10;
		lrand48_r(&buffer, &res);
		elem->back = res%2;
		xbt_mutex_acquire(mutex);
		if (elem->back)
			foo_prio_list_push_back(&mylist, elem);
		else
			foo_prio_list_push_front(&mylist, elem);
		check_list_prio(&mylist);
		xbt_mutex_release(mutex);
	}

	for (i = 0; i < M; i++)
	{
		lrand48_r(&buffer, &res);
		n = res%(M-i);

		xbt_mutex_acquire(mutex);
		for (elem  = foo_prio_list_begin(&mylist);
		     n--;
		     elem  = foo_prio_list_next(&mylist, elem))
			;
		foo_prio_list_erase(&mylist, elem);
		check_list_prio(&mylist);
		xbt_mutex_release(mutex);
	}

	return 0;
}

int master(int argc STARPU_ATTRIBUTE_UNUSED, char *argv[] STARPU_ATTRIBUTE_UNUSED)
{
	unsigned i;

	mutex = xbt_mutex_init();
	foo_prio_list_init(&mylist);

	for (i = 0; i < N; i++)
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
	xbt_cfg_set_int("contexts/stack-size", 128);
	MSG_create_environment(argv[1]);
	MSG_process_create("master", master, NULL, MSG_get_host_by_name(argv[2]));
	MSG_main();
	return 0;
}
