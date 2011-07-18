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

static void release_callback_event(void * e) {
  cl_event event = (cl_event)e;

  cl_command_queue cq = event->cq;

  /* Remove from command queue */
  if (cq != NULL) {
    /* Lock command queue */
    pthread_spin_lock(&cq->spin);

    /* Remove barrier if applicable */
    if (cq->barrier == event)
      cq->barrier = NULL;

    /* Remove from the list of out-of-order events */
    if (event->prev != NULL)
      event->prev->next = event->next;
    if (event->next != NULL)
      event->next->prev = event->prev;
    if (cq->events == event)
      cq->events = event->next;

    /* Unlock command queue */
    pthread_spin_unlock(&cq->spin);

    gc_entity_unstore(&cq);
  }

  /* Destruct object */
  //FIXME: we cannot release tag because it makes StarPU crash
  //starpu_tag_remove(event->id);
}

CL_API_ENTRY cl_int CL_API_CALL
soclReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
  if (event == NULL)
    return CL_INVALID_EVENT;

  gc_entity_release(event);

  return CL_SUCCESS;
}
