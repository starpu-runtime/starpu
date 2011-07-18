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

static void release_callback_memobject(void * e) {
  cl_mem mem = (cl_mem)e;

  /* Release references */
  gc_entity_unstore(&mem->context);

  //Delete this mem_object from the mem_object list
  mem_object_release(mem);

  /* Destruct object */
  starpu_data_unregister_no_coherency(mem->handle);

  if (!(mem->flags & CL_MEM_USE_HOST_PTR))
    free(mem->ptr);
}

CL_API_ENTRY cl_int CL_API_CALL
soclReleaseMemObject(cl_mem mem) CL_API_SUFFIX__VERSION_1_0
{
  gc_entity_release(mem);

  return CL_SUCCESS;
}
