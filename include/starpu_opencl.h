/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_OPENCL_H__
#define __STARPU_OPENCL_H__

#ifdef STARPU_USE_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <starpu_config.h>

#ifdef __cplusplus
extern "C" {
#endif

void starpu_opencl_display_error(const char *func, cl_int status);
#define STARPU_OPENCL_DISPLAY_ERROR(status) \
	starpu_opencl_display_error(__starpu_func__, status)

static inline void starpu_opencl_report_error(const char *func, cl_int status)
{
        starpu_opencl_display_error(func, status);
        assert(0);
}
#define STARPU_OPENCL_REPORT_ERROR(status)                              \
	starpu_opencl_display_error(__starpu_func__, status)

struct starpu_opencl_program {
        cl_program programs[STARPU_MAXOPENCLDEVS];
};

void starpu_opencl_get_context(int devid, cl_context *context);
void starpu_opencl_get_device(int devid, cl_device_id *device);
void starpu_opencl_get_queue(int devid, cl_command_queue *queue);
void starpu_opencl_get_current_context(cl_context *context);
void starpu_opencl_get_current_queue(cl_command_queue *queue);

int starpu_opencl_load_opencl_from_file(const char *source_file_name, struct starpu_opencl_program *opencl_programs,
					const char* build_options);
int starpu_opencl_load_opencl_from_string(const char *opencl_program_source, struct starpu_opencl_program *opencl_programs,
					  const char* build_options);
int starpu_opencl_unload_opencl(struct starpu_opencl_program *opencl_programs);

int starpu_opencl_load_kernel(cl_kernel *kernel, cl_command_queue *queue, struct starpu_opencl_program *opencl_programs, const char *kernel_name, int devid);
int starpu_opencl_release_kernel(cl_kernel kernel);

int starpu_opencl_collect_stats(cl_event event);


#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_OPENCL */
#endif /* __STARPU_OPENCL_H__ */

