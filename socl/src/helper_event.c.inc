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

/**
 * Create a new event
 *
 * Events have one-to-one relation with tag. Tag number is event ID
 */
static cl_event event_create(void) {
   static int id = 1;
   cl_event ev;
   ev = gc_entity_alloc(sizeof(struct _cl_event), release_callback_event);

   ev->next = NULL;
   ev->prev = NULL;
   ev->id = __sync_fetch_and_add(&id,1);
   ev->status = CL_SUBMITTED;
   ev->type = 0;
   ev->profiling_info = NULL;
   ev->cq = NULL;

   return ev;
}

