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

static void release_callback_command_queue(void * e) {
  cl_command_queue cq = (cl_command_queue)e;

  //Disable StarPU profiling if necessary
  if (cq->properties & CL_QUEUE_PROFILING_ENABLE) {
    profiling_queue_count -= 1;
    if (profiling_queue_count == 0)
      starpu_profiling_status_set(STARPU_PROFILING_DISABLE);
  }

  /* Release references */
  gc_entity_unstore(&cq->context);

  /* Destruct object */
  pthread_spin_destroy(&cq->spin);
  free(cq->events);
}


CL_API_ENTRY cl_int CL_API_CALL
soclReleaseCommandQueue(cl_command_queue cq) CL_API_SUFFIX__VERSION_1_0
{

  gc_entity_release(cq);

  return CL_SUCCESS;
}
