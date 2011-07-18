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

static void rk_task(void *data) {
   cl_kernel k = (cl_kernel)data;

   int range = starpu_worker_get_range();

   cl_int err = clReleaseKernel(k->cl_kernels[range]);
   if (err != CL_SUCCESS)
      DEBUG_CL("clReleaseKernel", err);
}

static void release_callback_kernel(void * e) {
  cl_kernel kernel = (cl_kernel)e;

  //Free args
  unsigned int j;
  for (j=0; j<kernel->arg_count; j++) {
    switch (kernel->arg_type[j]) {
      case Null:
        break;
      case Buffer:
        gc_entity_unstore((cl_mem*)&kernel->arg_value[j]);
        break;
      case Immediate:
        free(kernel->arg_value[j]);
        break;
    }
  }
  if (kernel->arg_size != NULL)
    free(kernel->arg_size);
  if (kernel->arg_value != NULL)
    free(kernel->arg_value);
  if (kernel->arg_type != NULL)
    free(kernel->arg_type);

  //Release real kernels...
  starpu_execute_on_each_worker(rk_task, kernel, STARPU_OPENCL);

  gc_entity_unstore(&kernel->program);

  free(kernel->kernel_name);
  free(kernel->cl_kernels);
  free(kernel->errcodes);
}

CL_API_ENTRY cl_int CL_API_CALL
soclReleaseKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
  if (kernel == NULL)
    return CL_INVALID_KERNEL;

  gc_entity_release(kernel);

  return CL_SUCCESS;
}
