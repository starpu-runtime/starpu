/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2012       Vincent Danjean
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

#include "gc.h"
#include "event.h"
#include "socl.h"

#include <stdlib.h>

/**
 * Garbage collection thread
 */

/* List of entities to be released */
static volatile entity gc_list = NULL;
static volatile entity entities = NULL;

/* Mutex and cond for release */
static starpu_pthread_mutex_t gc_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t  gc_cond = STARPU_PTHREAD_COND_INITIALIZER;

/* Set to 1 to stop release thread execution */
static volatile int gc_stop_required = 0;

#define GC_LOCK STARPU_PTHREAD_MUTEX_LOCK(&gc_mutex)
#define GC_UNLOCK { STARPU_PTHREAD_COND_SIGNAL(&gc_cond); \
                    STARPU_PTHREAD_MUTEX_UNLOCK(&gc_mutex);}
#define GC_UNLOCK_NO_SIGNAL STARPU_PTHREAD_MUTEX_UNLOCK(&gc_mutex)

/* Thread routine */
static void * gc_thread_routine(void *UNUSED(arg))
{
	GC_LOCK;

	do
	{
		/* Make a copy of the gc_list to allow callbacks to add things into it */
		entity rs = gc_list;
		gc_list = NULL;

		GC_UNLOCK_NO_SIGNAL;

		entity r = rs;
		while (r != NULL)
		{
			/* Call entity release callback */
			if (r->release_callback != NULL)
			{
				r->release_callback(r);
			}

			/* Release entity */
			entity next = r->next;
			free(r);

			r = next;
		}

		GC_LOCK;

		/* Check if new entities have been added */
		if (gc_list != NULL)
			continue;

		/* Stop if required */
		if (gc_stop_required)
		{
			GC_UNLOCK_NO_SIGNAL;
			break;
		}

		/* Otherwise we sleep */
		STARPU_PTHREAD_COND_WAIT(&gc_cond, &gc_mutex);
	} while (1);

	starpu_pthread_exit(NULL);
}

static starpu_pthread_t gc_thread;

/* Start garbage collection */
void gc_start(void)
{
	STARPU_PTHREAD_CREATE(&gc_thread, NULL, gc_thread_routine, NULL);
}

/* Stop garbage collection */
void gc_stop(void)
{
	GC_LOCK;

	gc_stop_required = 1;

	GC_UNLOCK;

	STARPU_PTHREAD_JOIN(gc_thread, NULL);
}

int gc_entity_release_ex(entity e, const char * DEBUG_PARAM(caller))
{
	DEBUG_MSG("[%s] Decrementing refcount of %s %p to ", caller, e->name, (void *)e);

	/* Decrement reference count */
	int refs = __sync_sub_and_fetch(&e->refs, 1);

	DEBUG_MSG_NOHEAD("%d\n", refs);

	assert(refs >= 0);

	if (refs != 0)
		return 0;

	DEBUG_MSG("[%s] Releasing %s %p\n", caller, e->name, (void *)e);

	GC_LOCK;

	/* Remove entity from the entities list */
	if (e->prev != NULL)
		e->prev->next = e->next;
	if (e->next != NULL)
		e->next->prev = e->prev;
	if (entities == e)
		entities = e->next;

	/* Put entity in the release queue */
	e->next = gc_list;
	gc_list = e;

	GC_UNLOCK;

	return 1;
}

/**
 * Initialize entity
 */
void gc_entity_init(void *arg, void (*release_callback)(void*), char * name)
{
	DEBUG_MSG("Initializing entity %p (%s)\n", arg, name);

	struct entity * e = (entity)arg;

	e->dispatch = &socl_master_dispatch;
	e->refs = 1;
	e->release_callback = release_callback;
	e->prev = NULL;
	e->name = name;

	GC_LOCK;

	e->next = entities;
	if (entities != NULL)
		entities->prev = e;
	entities = e;

	GC_UNLOCK_NO_SIGNAL;
}

/**
 * Allocate and initialize entity
 */
void * gc_entity_alloc(unsigned int size, void (*release_callback)(void*), char * name)
{
	void * e = malloc(size);
	gc_entity_init(e, release_callback, name);
	return e;
}

/** Retain entity */
void gc_entity_retain_ex(void *arg, const char * DEBUG_PARAM(caller))
{
	struct entity * e = (entity)arg;

#ifdef DEBUG
	int refs =
#endif
		__sync_add_and_fetch(&e->refs, 1);

	DEBUG_MSG("[%s] Incrementing refcount of %s %p to %d\n", caller, e->name, e, refs);
}

int gc_active_entity_count(void)
{
	int i = 0;

	entity e = entities;
	while (e != NULL)
	{
		i++;
		e = e->next;
	}

	return i;
}

void gc_print_remaining_entities(void)
{
	DEBUG_MSG("Remaining entities:\n");

	GC_LOCK;

	entity e = entities;
	while (e != NULL)
	{
		DEBUG_MSG("  - %s %p\n", e->name, (void *)e);
		e = e->next;
	}

	GC_UNLOCK;
}

#undef GC_LOCK
#undef GC_UNLOCK
#undef GC_UNLOCK_NO_SIGNAL
