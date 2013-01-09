/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011 University of Bordeaux
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

int event_unique_id() {
   static int id = 1;

   return __sync_fetch_and_add(&id,1);
}

/**
 * Create a new event
 *
 * Events have one-to-one relation with tag. Tag number is event ID
 */
cl_event event_create(void) {
   cl_event ev;
   ev = gc_entity_alloc(sizeof(struct _cl_event), release_callback_event, "event");

   ev->id = event_unique_id();
   ev->status = CL_SUBMITTED;
   ev->command = NULL;
   ev->profiling_info = NULL;
   ev->cq = NULL;

   return ev;
}

static void release_callback_event(void * e) {
  cl_event event = (cl_event)e;

  gc_entity_unstore(&event->cq);

  /* Destruct object */
  //FIXME
  //starpu_tag_remove(event->id);
  if (event->profiling_info != NULL) {
    free(event->profiling_info);
    event->profiling_info = NULL;
  }
}

