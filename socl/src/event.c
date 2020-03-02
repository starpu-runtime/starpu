/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "socl.h"
#include "event.h"
#include "gc.h"

static void release_callback_event(void * e);

int event_unique_id()
{
	static int id = 1;

	return __sync_fetch_and_add(&id,1);
}

/**
 * Create a new event
 *
 * Events have one-to-one relation with tag. Tag number is event ID
 */
cl_event event_create(void)
{
	cl_event ev;
	ev = gc_entity_alloc(sizeof(struct _cl_event), release_callback_event, "event");

	ev->id = event_unique_id();
	ev->status = CL_SUBMITTED;
	ev->command = NULL;
	ev->prof_queued = 0L;
	ev->prof_submit = 0L;
	ev->prof_start = 0L;
	ev->prof_end = 0L;
	ev->cq = NULL;

	return ev;
}

void event_complete(cl_event ev)
{
	ev->status = CL_COMPLETE;

	ev->prof_end = _socl_nanotime();

	/* Trigger the tag associated to the command event */
	DEBUG_MSG("Trigger event %d\n", ev->id);
	starpu_tag_notify_from_apps(ev->id);
}

static void release_callback_event(void * e)
{
	cl_event event = (cl_event)e;

	gc_entity_unstore(&event->cq);

	/* Destruct object */
	//FIXME
	//starpu_tag_remove(event->id);
}
