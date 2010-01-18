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

#include <pthread.h>
#include <core/workers.h>

#define NMAXHOOKS	16

struct progression_hook {
	unsigned (*func)(void *arg);
	void *arg;
	unsigned active;
};

/* protect the hook table */
static pthread_mutex_t progression_hook_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct progression_hook hooks[NMAXHOOKS] = {{NULL, NULL, 0}};

int starpu_register_progression_hook(unsigned (*func)(void *arg), void *arg)
{
	int hook;
	pthread_mutex_lock(&progression_hook_mutex);
	for (hook = 0; hook < NMAXHOOKS; hook++)
	{
		if (!hooks[hook].active)
		{
			/* We found an empty slot */
			hooks[hook].func = func;
			hooks[hook].arg = arg;
			hooks[hook].active = 1;

			pthread_mutex_unlock(&progression_hook_mutex);
			
			return hook;
		}
	}

	pthread_mutex_unlock(&progression_hook_mutex);

	starpu_wake_all_blocked_workers();

	/* We could not find an empty slot */
	return -1;
}

void starpu_deregister_progression_hook(int hook_id)
{
	pthread_mutex_lock(&progression_hook_mutex);
	hooks[hook_id].active = 0;
	pthread_mutex_unlock(&progression_hook_mutex);
}

unsigned _starpu_execute_registered_progression_hooks(void)
{
	/* By default, it is possible to block, but if some progression hooks
	 * requires that it's not blocking, we disable blocking. */
	unsigned may_block = 1;

	unsigned hook;
	for (hook = 0; hook < NMAXHOOKS; hook++)
	{
		unsigned active;

		pthread_mutex_lock(&progression_hook_mutex);
		active = hooks[hook].active;
		pthread_mutex_unlock(&progression_hook_mutex);

		unsigned may_block_hook = 1;

		if (active)
			may_block_hook = hooks[hook].func(hooks[hook].arg);

		/* As soon as one hook tells that the driver cannot be
		 * blocking, we don't allow it. */
		if (!may_block_hook)
			may_block = 0;
	}

	return may_block;
}
