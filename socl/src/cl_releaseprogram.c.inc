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

static void rp_task(void *data) {
   struct _cl_program *d = (struct _cl_program*)data;

   int range = starpu_worker_get_range();

   cl_int err = clReleaseProgram(d->cl_programs[range]);
   if (err != CL_SUCCESS)
      DEBUG_CL("clReleaseProgram", err);
}

static void release_callback_program(void * e) {
  cl_program program = (cl_program)e;

  /* Destruct object */
  starpu_execute_on_each_worker(rp_task, program, STARPU_OPENCL);

  /* Release references */
  gc_entity_unstore(&program->context);

  free(program->cl_programs);

  if (program->options != NULL)
    free(program->options);
}

CL_API_ENTRY cl_int CL_API_CALL
soclReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
  if (program == NULL)
    return CL_INVALID_PROGRAM;

  gc_entity_release(program);

  return CL_SUCCESS;
}
