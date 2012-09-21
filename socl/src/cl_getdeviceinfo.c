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
#include "getinfo.h"


CL_API_ENTRY cl_int CL_API_CALL
soclGetDeviceInfo(cl_device_id    device,
                cl_device_info  param_name, 
                size_t          param_value_size, 
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 
{

   //FIXME: we do not check if the device is valid
   /* if (device != &socl_virtual_device && device is not a valid StarPU worker identifier)
      return CL_INVALID_DEVICE;*/

  int devid = device->device_id;

  cl_device_id dev;
  starpu_opencl_get_device(devid, &dev);

  int ret = clGetDeviceInfo(dev, param_name, param_value_size, param_value, param_value_size_ret);

  return ret;

}
