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
enum _starpu_prof_tool_event
{
	_starpu_prof_tool_event_none = 0,
	_starpu_prof_tool_event_init,
	_starpu_prof_tool_event_terminate,
	_starpu_prof_tool_event_init_begin,
	_starpu_prof_tool_event_init_end,

	_starpu_prof_tool_event_driver_init,
	_starpu_prof_tool_event_driver_deinit,
	_starpu_prof_tool_event_driver_init_start,
	_starpu_prof_tool_event_driver_init_end,
	_starpu_prof_tool_event_start_cpu_exec,
	_starpu_prof_tool_event_end_cpu_exec,
	_starpu_prof_tool_event_start_gpu_exec,
	_starpu_prof_tool_event_end_gpu_exec,
	_starpu_prof_tool_event_start_transfer,
	_starpu_prof_tool_event_end_transfer,

	_starpu_prof_tool_event_user_start,
	_starpu_prof_tool_event_user_end
};

enum  _starpu_prof_tool_command
{
	_starpu_reg = 0,
	_starpu_toggle = 1,
	_starpu_toggle_per_thread = 2
};

enum _starpu_driver_type
{
	_starpu_driver_cpu,
	_starpu_driver_gpu
};

/*
  First argument:
  General information
*/
struct _starpu_prof_tool_info
{
	struct starpu_conf *conf;
	enum _starpu_prof_tool_event event_type;
	unsigned int starpu_version[3];
	int thread_id;
	int worker_id;

	int device_number;
	enum _starpu_driver_type driver_type; // not sure

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

/*
  Second argument:
  Event info
*/
union _starpu_prof_tool_event_info
{
	enum _starpu_prof_tool_event event_type;
	/*   starpu_data_event_info data_event;
	     starpu_launch_event_info launch_event;
	     starpu_other_event_info other_event;*/
};

/*
  Third argument:
  API info
*/
struct _starpu_prof_tool_api_info
{
	/*acc_device_api device_api;
	  int valid_bytes;
	  acc_device_t device_type;
	  int vendor;
	  const void* device_handle;
	  const void* context_handle;
	  const void* async_handle;*/
};

/*******************************************************************************
 * Callback signature
 *******************************************************************************/

typedef void (*_starpu_prof_tool_cb_func)(struct _starpu_prof_tool_info*, union _starpu_prof_tool_event_info*, struct _starpu_prof_tool_api_info*);

/*
  This function must be called by external tools that want
  to use the callbacks
*/
void _starpu_prof_tool_library_init(void);

/*
  Register / unregister events
*/
typedef void (*_starpu_prof_tool_entry_register_func)(enum _starpu_prof_tool_event event_type, _starpu_prof_tool_cb_func cb, enum _starpu_prof_tool_command info);

/*
  This function must be implemented by external tools that want
  to use the callbacks
  TODO: both?
  TODO: removed the lookup argument
*/
void starpu_prof_tool_library_register(_starpu_prof_tool_entry_register_func reg, _starpu_prof_tool_entry_register_func unreg);
typedef void (*_starpu_prof_tool_entry_func)(_starpu_prof_tool_entry_register_func reg, _starpu_prof_tool_entry_register_func unreg);

/* The events themselves.
   This structure can be built by the preprocessor, but we decided
   to list the function pointers explicitly for readability purpose.
*/
struct _starpu_prof_tool_callbacks
{
	_starpu_prof_tool_cb_func starpu_prof_tool_event_init;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_terminate;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_init_begin;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_init_end;

	_starpu_prof_tool_cb_func starpu_prof_tool_event_driver_init;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_driver_deinit;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_driver_init_start;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_driver_init_end;

	_starpu_prof_tool_cb_func starpu_prof_tool_event_start_cpu_exec;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_end_cpu_exec;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_start_gpu_exec;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_end_gpu_exec;

	_starpu_prof_tool_cb_func starpu_prof_tool_event_start_transfer;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_end_transfer;

	_starpu_prof_tool_cb_func starpu_prof_tool_event_user_start;
	_starpu_prof_tool_cb_func starpu_prof_tool_event_user_end;
};

extern struct _starpu_prof_tool_callbacks starpu_prof_tool_callbacks;

/*******************************************************************************
 * Functions used by the callbacks
 *******************************************************************************/
struct _starpu_prof_tool_info _starpu_prof_tool_get_info(enum _starpu_prof_tool_event, int, enum _starpu_driver_type, unsigned int, /*_starpu_cl_func_t*/ void*);
struct _starpu_prof_tool_info _starpu_prof_tool_get_info_d(enum _starpu_prof_tool_event, int, enum _starpu_driver_type, unsigned, unsigned, unsigned /* void*: can be added later if necessary */);
struct _starpu_prof_tool_info _starpu_prof_tool_get_info_init(enum _starpu_prof_tool_event, int, enum _starpu_driver_type, struct starpu_conf*);

/*******************************************************************************
 * Initialization and cleanup
 *******************************************************************************/
int _starpu_prof_tool_try_load();
void _starpu_prof_tool_unload();

#ifdef __cplusplus
}
#endif

#if 0
acc_prof_info pi = acc_get_prof_info(event_type, device_num);
acc_event_info ei = acc_get_launch_event_info(event_type);
acc_api_info ai = acc_get_api_info(device_num);
#endif

#endif  // _STARPU_CALLBACKS_H_:x
