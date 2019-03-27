/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019                                     Inria
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

#ifndef __STARPU_PERF_STEERING_H__
#define __STARPU_PERF_STEERING_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Perf_Steering Perf_Steering
   @{
*/

enum starpu_perf_knob_scope
{
	starpu_perf_knob_scope_undefined     = 0,
	starpu_perf_knob_scope_global        = 1,
	starpu_perf_knob_scope_per_worker    = 3,
	starpu_perf_knob_scope_per_scheduler = 5
};

enum starpu_perf_knob_type
{
	starpu_perf_knob_type_undefined = 0,
	starpu_perf_knob_type_int32     = 1,
	starpu_perf_knob_type_int64     = 2,
	starpu_perf_knob_type_float     = 3,
	starpu_perf_knob_type_double    = 4
};

int starpu_perf_knob_scope_name_to_id(const char *name);
const char *starpu_perf_knob_scope_id_to_name(enum starpu_perf_knob_scope scope);

int starpu_perf_knob_type_name_to_id(const char *name);
const char *starpu_perf_knob_type_id_to_name(enum starpu_perf_knob_type type);

int starpu_perf_knob_nb(enum starpu_perf_knob_scope scope);
int starpu_perf_knob_name_to_id(enum starpu_perf_knob_scope scope, const char *name);
int starpu_perf_knob_nth_to_id(enum starpu_perf_knob_scope scope, int nth);
const char *starpu_perf_knob_id_to_name(int id);
int starpu_perf_knob_get_type_id(int id);
const char *starpu_perf_knob_get_help_string(int id);

void starpu_perf_knob_list_avail(enum starpu_perf_knob_scope scope);
void starpu_perf_knob_list_all_avail(enum starpu_perf_knob_scope scope);

int32_t starpu_perf_knob_get_global_int32_value (const int knob_id);
int64_t starpu_perf_knob_get_global_int64_value (const int knob_id);
float   starpu_perf_knob_get_global_float_value (const int knob_id);
double  starpu_perf_knob_get_global_double_value(const int knob_id);

void starpu_perf_knob_set_global_int32_value (const int knob_id, int32_t new_value);
void starpu_perf_knob_set_global_int64_value (const int knob_id, int64_t new_value);
void starpu_perf_knob_set_global_float_value (const int knob_id, float   new_value);
void starpu_perf_knob_set_global_double_value(const int knob_id, double  new_value);


int32_t starpu_perf_knob_get_per_worker_int32_value (const int knob_id, unsigned workerid);
int64_t starpu_perf_knob_get_per_worker_int64_value (const int knob_id, unsigned workerid);
float   starpu_perf_knob_get_per_worker_float_value (const int knob_id, unsigned workerid);
double  starpu_perf_knob_get_per_worker_double_value(const int knob_id, unsigned workerid);

void starpu_perf_knob_set_per_worker_int32_value (const int knob_id, unsigned workerid, int32_t new_value);
void starpu_perf_knob_set_per_worker_int64_value (const int knob_id, unsigned workerid, int64_t new_value);
void starpu_perf_knob_set_per_worker_float_value (const int knob_id, unsigned workerid, float   new_value);
void starpu_perf_knob_set_per_worker_double_value(const int knob_id, unsigned workerid, double  new_value);


int32_t starpu_perf_knob_get_per_scheduler_int32_value (const int knob_id, const char * sched_policy_name);
int64_t starpu_perf_knob_get_per_scheduler_int64_value (const int knob_id, const char * sched_policy_name);
float   starpu_perf_knob_get_per_scheduler_float_value (const int knob_id, const char * sched_policy_name);
double  starpu_perf_knob_get_per_scheduler_double_value(const int knob_id, const char * sched_policy_name);

void starpu_perf_knob_set_per_scheduler_int32_value (const int knob_id, const char * sched_policy_name, int32_t new_value);
void starpu_perf_knob_set_per_scheduler_int64_value (const int knob_id, const char * sched_policy_name, int64_t new_value);
void starpu_perf_knob_set_per_scheduler_float_value (const int knob_id, const char * sched_policy_name, float   new_value);
void starpu_perf_knob_set_per_scheduler_double_value(const int knob_id, const char * sched_policy_name, double  new_value);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERF_STEERING_H__ */
