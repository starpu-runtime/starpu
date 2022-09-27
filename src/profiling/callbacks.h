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

#include <starpu.h>
#include <common/config.h>

#define STARPU_PROF_TOOL_ENV_VAR "STARPU_PROF_TOOL"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * Callback signature
 *******************************************************************************/

/*
  This function must be implemented by external tools that want
  to use the callbacks
  TODO: both?
  TODO: removed the lookup argument
*/
void starpu_prof_tool_library_register(starpu_prof_tool_entry_register_func reg, starpu_prof_tool_entry_register_func unreg);

/* The events themselves.
   This structure can be built by the preprocessor, but we decided
   to list the function pointers explicitly for readability purpose.
*/
struct _starpu_prof_tool_callbacks
{
	starpu_prof_tool_cb_func starpu_prof_tool_event_init;
	starpu_prof_tool_cb_func starpu_prof_tool_event_terminate;
	starpu_prof_tool_cb_func starpu_prof_tool_event_init_begin;
	starpu_prof_tool_cb_func starpu_prof_tool_event_init_end;

	starpu_prof_tool_cb_func starpu_prof_tool_event_driver_init;
	starpu_prof_tool_cb_func starpu_prof_tool_event_driver_deinit;
	starpu_prof_tool_cb_func starpu_prof_tool_event_driver_init_start;
	starpu_prof_tool_cb_func starpu_prof_tool_event_driver_init_end;

	starpu_prof_tool_cb_func starpu_prof_tool_event_start_cpu_exec;
	starpu_prof_tool_cb_func starpu_prof_tool_event_end_cpu_exec;
	starpu_prof_tool_cb_func starpu_prof_tool_event_start_gpu_exec;
	starpu_prof_tool_cb_func starpu_prof_tool_event_end_gpu_exec;

	starpu_prof_tool_cb_func starpu_prof_tool_event_start_transfer;
	starpu_prof_tool_cb_func starpu_prof_tool_event_end_transfer;

	starpu_prof_tool_cb_func starpu_prof_tool_event_user_start;
	starpu_prof_tool_cb_func starpu_prof_tool_event_user_end;
};

extern struct _starpu_prof_tool_callbacks starpu_prof_tool_callbacks;

/*******************************************************************************
 * Functions used by the callbacks
 *******************************************************************************/
struct starpu_prof_tool_info _starpu_prof_tool_get_info(enum starpu_prof_tool_event, int, enum starpu_prof_tool_driver_type, unsigned int, /*_starpu_cl_func_t*/ void*);
struct starpu_prof_tool_info _starpu_prof_tool_get_info_d(enum starpu_prof_tool_event, int, enum starpu_prof_tool_driver_type, unsigned, unsigned, unsigned /* void*: can be added later if necessary */);
struct starpu_prof_tool_info _starpu_prof_tool_get_info_init(enum starpu_prof_tool_event, int, enum starpu_prof_tool_driver_type, struct starpu_conf*);

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
