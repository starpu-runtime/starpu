/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2022  Camille Coti
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

#ifndef _STARPU_CALLBACKS_H_
#define _STARPU_CALLBACKS_H_

#include "config.h"

#define STARPU_PROF_TOOL_ENV_VAR "STARPU_PROF_TOOL"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * Datatypes
 *******************************************************************************/

/*
  Event type
*/

typedef enum
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
} starpu_prof_tool_event_t;

typedef enum
{
	starpu_reg = 0,
	starpu_toggle = 1,
	starpu_toggle_per_thread = 2
} starpu_prof_tool_command_t;

typedef enum
{
	starpu_driver_cpu,
	starpu_driver_gpu
} starpu_driver_type_t;

/*
  First argument:
  General information
*/

typedef struct
{
	struct starpu_conf *conf;
	starpu_prof_tool_event_t event_type;
	unsigned int starpu_version[3];
	int thread_id;
	int worker_id;

	int device_number;
	starpu_driver_type_t driver_type; // not sure

	unsigned memnode;
	unsigned bytes_to_transfer;
	unsigned bytes_transfered;

	void* fun_ptr;  /* NULL when not relevant (driver init etc) */

	/*    int valid_bytes;
	      int version;
	      starpu_device_t device_type;
	      int device_number;
	      ssize_t async;
	      ssize_t async_queue;
	      const char* src_file;
	      const char* func_name;
	      int line_no, end_line_no;
	      int func_line_no, func_end_line_no;*/
} starpu_prof_tool_info_t;

/*
  Second argument:
  Event info
*/
typedef union
{
	starpu_prof_tool_event_t event_type;
	/*   starpu_data_event_info data_event;
	     starpu_launch_event_info launch_event;
	     starpu_other_event_info other_event;*/
} starpu_prof_tool_event_info_t;

/*
  Third argument:
  API info
*/
typedef struct
{
	/*acc_device_api device_api;
	  int valid_bytes;
	  acc_device_t device_type;
	  int vendor;
	  const void* device_handle;
	  const void* context_handle;
	  const void* async_handle;*/
} starpu_prof_tool_api_info_t;

/*******************************************************************************
 * Callback signature
 *******************************************************************************/

typedef void (*starpu_prof_tool_cb_func_t)(starpu_prof_tool_info_t*, starpu_prof_tool_event_info_t*, starpu_prof_tool_api_info_t*);

/*
  This function must be called by external tools that want
  to use the callbacks
*/
void starpu_prof_tool_library_init(void);

/*
  Register / unregister events
*/
typedef void (*starpu_prof_tool_entry_t)(starpu_prof_tool_event_t event_type, starpu_prof_tool_cb_func_t cb, starpu_prof_tool_command_t info);

/*
  This function must be implemented by external tools that want
  to use the callbacks
  TODO: both?
  TODO: removed the lookup argument
*/
void starpu_prof_tool_library_register(starpu_prof_tool_entry_t reg, starpu_prof_tool_entry_t unreg);
typedef void (*starpu_prof_tool_entry_func_t)(starpu_prof_tool_entry_t reg, starpu_prof_tool_entry_t unreg);

/* The events themselves.
   This structure can be built by the preprocessor, but we decided
   to list the function pointers explicitly for readability purpose.
*/
typedef struct
{
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_init;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_terminate;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_init_begin;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_init_end;

	starpu_prof_tool_cb_func_t starpu_prof_tool_event_driver_init;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_driver_deinit;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_driver_init_start;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_driver_init_end;

	starpu_prof_tool_cb_func_t starpu_prof_tool_event_start_cpu_exec;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_end_cpu_exec;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_start_gpu_exec;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_end_gpu_exec;

	starpu_prof_tool_cb_func_t starpu_prof_tool_event_start_transfer;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_end_transfer;

	starpu_prof_tool_cb_func_t starpu_prof_tool_event_user_start;
	starpu_prof_tool_cb_func_t starpu_prof_tool_event_user_end;
} starpu_prof_tool_callbacks_t;

extern starpu_prof_tool_callbacks_t starpu_prof_tool_callbacks;

#define STARPU_NB_CALLBACKS   17
extern starpu_prof_tool_cb_func_t* starpu_prof_tool_callback_map[STARPU_NB_CALLBACKS];

/*******************************************************************************
 * Functions used by the callbacks
 *******************************************************************************/
starpu_prof_tool_info_t starpu_prof_tool_get_info(starpu_prof_tool_event_t, int, starpu_driver_type_t, unsigned int, /*_starpu_cl_func_t*/ void*);
starpu_prof_tool_info_t starpu_prof_tool_get_info_d(starpu_prof_tool_event_t, int, starpu_driver_type_t, unsigned, unsigned, unsigned /* void*: can be added later if necessary */);
starpu_prof_tool_info_t starpu_prof_tool_get_info_init(starpu_prof_tool_event_t, int, starpu_driver_type_t, struct starpu_conf*);

/*******************************************************************************
 * Initialization and cleanup
 *******************************************************************************/
int starpu_prof_tool_try_load();
void starpu_prof_tool_unload();

#ifdef __cplusplus
}
#endif

#if 0
acc_prof_info pi = acc_get_prof_info(event_type, device_num);
acc_event_info ei = acc_get_launch_event_info(event_type);
acc_api_info ai = acc_get_api_info(device_num);
#endif

#endif  // _STARPU_CALLBACKS_H_:x
