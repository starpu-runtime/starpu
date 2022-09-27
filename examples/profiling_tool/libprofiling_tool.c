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

#include <starpu.h>
#include <starpu_profiling_tool.h>

void myfunction_cb(struct starpu_prof_tool_info *prof_info, union starpu_prof_tool_event_info *event_info, struct starpu_prof_tool_api_info *api_info)
{
	if (NULL != prof_info)
	{
		printf("CALLBACK CALLED %d\n", prof_info->event_type);
	}
	else
	{
		printf("CALLBACK CALLED NULL INFO\n");
        return;
	}

	switch (prof_info->event_type)
	{
	case starpu_prof_tool_event_driver_init:
		printf("init driver\n");
		break;
	case starpu_prof_tool_event_driver_init_start:
		printf("begin init driver\n");
		break;
	case starpu_prof_tool_event_driver_init_end:
		printf("end init driver\n");
		break;
	case starpu_prof_tool_event_start_cpu_exec:
		printf("Start exec fun %p on device %d\n", prof_info->fun_ptr, prof_info->device_number);
		break;
	case starpu_prof_tool_event_end_cpu_exec:
		printf("End exec fun %p on device %d\n", prof_info->fun_ptr, prof_info->device_number);
		break;
	case starpu_prof_tool_event_start_transfer:
		printf("Start transfer on memnode %ud\n", prof_info->memnode);
		break;
	case starpu_prof_tool_event_end_transfer:
		printf("End transfer on memnode %ud\n", prof_info->memnode);
		break;
	default:
		printf("Unknown callback %d\n",  prof_info->event_type);
		break;
	}
}

/* Mandatory */
void starpu_prof_tool_library_register(starpu_prof_tool_entry_register_func reg, starpu_prof_tool_entry_register_func unreg)
{
	enum  starpu_prof_tool_command info = 0;
	reg(starpu_prof_tool_event_driver_init, &myfunction_cb, info);
	reg(starpu_prof_tool_event_driver_init_start, &myfunction_cb, info);
	reg(starpu_prof_tool_event_driver_init_end, &myfunction_cb, info);
	reg(starpu_prof_tool_event_start_cpu_exec, &myfunction_cb, info);
	reg(starpu_prof_tool_event_end_cpu_exec, &myfunction_cb, info);
	reg(starpu_prof_tool_event_start_transfer, &myfunction_cb, info);
	reg(starpu_prof_tool_event_end_transfer, &myfunction_cb, info);

	printf("REGISTER LIBRARY\n");
}

