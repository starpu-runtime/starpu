/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2022-2023  Camille Coti
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

#ifndef __STARPU_PROFILING_TOOL_H__
#define __STARPU_PROFILING_TOOL_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Profiling_Tool Profiling Tool
   @{
*/

/**
   Event type
*/
enum starpu_prof_tool_event
{
	starpu_prof_tool_event_none = 0,
	starpu_prof_tool_event_init,
	starpu_prof_tool_event_terminate,
	starpu_prof_tool_event_init_begin,
	starpu_prof_tool_event_init_end,

	starpu_prof_tool_event_driver_init,
	starpu_prof_tool_event_driver_deinit,
	starpu_prof_tool_event_driver_init_start,
	starpu_prof_tool_event_driver_init_end,
	starpu_prof_tool_event_start_cpu_exec,
	starpu_prof_tool_event_end_cpu_exec,
	starpu_prof_tool_event_start_gpu_exec,
	starpu_prof_tool_event_end_gpu_exec,
	starpu_prof_tool_event_start_transfer,
	starpu_prof_tool_event_end_transfer,

	starpu_prof_tool_event_user_start,
	starpu_prof_tool_event_user_end
};

/**
   todo
*/
enum starpu_prof_tool_driver_type
{
	starpu_prof_tool_driver_cpu,
	starpu_prof_tool_driver_gpu,
	starpu_prof_tool_driver_hip,
	starpu_prof_tool_driver_ocl
};

/**
   todo
*/
enum starpu_prof_tool_command
{
	starpu_prof_tool_command_reg = 0,
	starpu_prof_tool_command_toggle = 1,
	starpu_prof_tool_command_toggle_per_thread = 2
};

/**
   General information
*/
struct starpu_prof_tool_info
{
	struct starpu_conf *conf;
	enum starpu_prof_tool_event event_type;
	unsigned int starpu_version[3];
	int thread_id;
	int worker_id;

	const char *task_name;
	const char *model_name;

	int device_number;
	enum starpu_prof_tool_driver_type driver_type; // not sure

	unsigned memnode;
	unsigned bytes_to_transfer;
	unsigned bytes_transfered;

	void* fun_ptr;  /* NULL when not relevant (driver init etc) */

	/*    int valid_bytes;
	      int version;
	      starpu_device_t device_type;
	      int device_number;
	      starpu_ssize_t async;
	      starpu_ssize_t async_queue;
	      const char* src_file;
	      const char* func_name;
	      int line_no, end_line_no;
	      int func_line_no, func_end_line_no;*/
};

/**
   Event info
*/
union starpu_prof_tool_event_info
{
	enum starpu_prof_tool_event event_type;
	/*   starpu_data_event_info data_event;
	     starpu_launch_event_info launch_event;
	     starpu_other_event_info other_event;*/
};

/**
   API info
*/
struct starpu_prof_tool_api_info
{
	/*acc_device_api device_api;
	  int valid_bytes;
	  acc_device_t device_type;
	  int vendor;
	  const void* device_handle;
	  const void* context_handle;
	  const void* async_handle;*/
};

typedef void (*starpu_prof_tool_cb_func)(struct starpu_prof_tool_info*, union starpu_prof_tool_event_info*, struct starpu_prof_tool_api_info*);

/**
   Register / unregister events
*/
typedef void (*starpu_prof_tool_entry_register_func)(enum starpu_prof_tool_event event_type, starpu_prof_tool_cb_func cb, enum starpu_prof_tool_command info);

/**
   In order to use the StarPU profiling interface, a tool must implement the
   starpu_prof_tool_library_register function, through which StarPU initializes the tool.
*/
extern void starpu_prof_tool_library_register(starpu_prof_tool_entry_register_func reg, starpu_prof_tool_entry_register_func unreg);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PROFILING_TOOL_H__ */
