/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
extern "C" {
#endif

/**
   @defgroup API_Perf_Steering Performance Steering Knobs
   @brief This section describes the interface to access performance steering counters.
   @{
*/

/**
   @name API
   \anchor PM_API
   @{
*/

/**
   Enum of all possible performance knob scopes.
 */
enum starpu_perf_knob_scope
{
	starpu_perf_knob_scope_undefined     = 0, /**< undefined scope */
	starpu_perf_knob_scope_global	     = 1, /**< global scope */
	starpu_perf_knob_scope_per_worker    = 3, /**< per-worker scope */
	starpu_perf_knob_scope_per_scheduler = 5  /**< per-scheduler scope */
};

/**
   Enum of all possible performance knob value type.
 */
enum starpu_perf_knob_type
{
	starpu_perf_knob_type_undefined = 0, /**< underfined value type */
	starpu_perf_knob_type_int32	= 1, /**< signed 32-bit integer value */
	starpu_perf_knob_type_int64	= 2, /**< signed 64-bit integer value */
	starpu_perf_knob_type_float	= 3, /**< 32-bit single precision floating-point value */
	starpu_perf_knob_type_double	= 4  /**< 64-bit double precision floating-point value */
};

/**
   Translate scope name constant string to scope id.
*/
int starpu_perf_knob_scope_name_to_id(const char *name);

/**
   Translate scope id to scope name constant string.
*/
const char *starpu_perf_knob_scope_id_to_name(enum starpu_perf_knob_scope scope);

/**
   Translate type name constant string to type id.
*/
int starpu_perf_knob_type_name_to_id(const char *name);

/**
   Translate type id to type name constant string.
*/
const char *starpu_perf_knob_type_id_to_name(enum starpu_perf_knob_type type);

/**
   Return the number of performance steering knobs for the given scope.
*/
int starpu_perf_knob_nb(enum starpu_perf_knob_scope scope);

/**
   Translate a performance knob name to its id.
*/
int starpu_perf_knob_name_to_id(enum starpu_perf_knob_scope scope, const char *name);

/**
   Translate a performance knob name to its id.
*/
int starpu_perf_knob_nth_to_id(enum starpu_perf_knob_scope scope, int nth);

/**
   Translate a performance knob rank in its scope to its knob id.
*/
const char *starpu_perf_knob_id_to_name(int id);

/**
   Translate a knob id to its name constant string.
*/
int starpu_perf_knob_get_type_id(int id);

/**
   Return the knob's help string.
*/
const char *starpu_perf_knob_get_help_string(int id);

/**
   Display the list of knobs defined in the given scope.
*/
void starpu_perf_knob_list_avail(enum starpu_perf_knob_scope scope);

/**
   Display the list of knobs defined in all scopes.
*/
void starpu_perf_knob_list_all_avail(void);

/**
   Get knob value for Global scope.
*/
int32_t starpu_perf_knob_get_global_int32_value(const int knob_id);

/**
   Get knob value for Global scope.
*/
int64_t starpu_perf_knob_get_global_int64_value(const int knob_id);

/**
   Get knob value for Global scope.
*/
float starpu_perf_knob_get_global_float_value(const int knob_id);

/**
   Get knob value for Global scope.
*/
double starpu_perf_knob_get_global_double_value(const int knob_id);

/**
   Set int32 knob value for Global scope.
*/
void starpu_perf_knob_set_global_int32_value(const int knob_id, int32_t new_value);

/**
   Set int64 knob value for Global scope.
*/
void starpu_perf_knob_set_global_int64_value(const int knob_id, int64_t new_value);

/**
   Set float knob value for Global scope.
*/
void starpu_perf_knob_set_global_float_value(const int knob_id, float new_value);

/**
   Set double knob value for Global scope.
*/
void starpu_perf_knob_set_global_double_value(const int knob_id, double new_value);

/**
   Get int32 value for Per_worker scope.
*/
int32_t starpu_perf_knob_get_per_worker_int32_value(const int knob_id, unsigned workerid);

/**
   Get int64 value for Per_worker scope.
*/
int64_t starpu_perf_knob_get_per_worker_int64_value(const int knob_id, unsigned workerid);

/**
   Get float value for Per_worker scope.
*/
float starpu_perf_knob_get_per_worker_float_value(const int knob_id, unsigned workerid);

/**
   Get double value for Per_worker scope.
*/
double starpu_perf_knob_get_per_worker_double_value(const int knob_id, unsigned workerid);

/**
   Set int32 value for Per_worker scope.
*/
void starpu_perf_knob_set_per_worker_int32_value(const int knob_id, unsigned workerid, int32_t new_value);

/**
   Set int64 value for Per_worker scope.
*/
void starpu_perf_knob_set_per_worker_int64_value(const int knob_id, unsigned workerid, int64_t new_value);

/**
   Set float value for Per_worker scope.
*/
void starpu_perf_knob_set_per_worker_float_value(const int knob_id, unsigned workerid, float new_value);

/**
   Set double value for Per_worker scope.
*/
void starpu_perf_knob_set_per_worker_double_value(const int knob_id, unsigned workerid, double new_value);

/**
   Get int32 value for per_scheduler scope.
*/
int32_t starpu_perf_knob_get_per_scheduler_int32_value(const int knob_id, const char *sched_policy_name);

/**
   Get int64 value for per_scheduler scope.
*/
int64_t starpu_perf_knob_get_per_scheduler_int64_value(const int knob_id, const char *sched_policy_name);

/**
   Get float value for per_scheduler scope.
*/
float starpu_perf_knob_get_per_scheduler_float_value(const int knob_id, const char *sched_policy_name);

/**
   Get double value for per_scheduler scope.
*/
double starpu_perf_knob_get_per_scheduler_double_value(const int knob_id, const char *sched_policy_name);

/**
   Set int32 value for per_scheduler scope.
*/
void starpu_perf_knob_set_per_scheduler_int32_value(const int knob_id, const char *sched_policy_name, int32_t new_value);

/**
   Set int64 value for per_scheduler scope.
*/
void starpu_perf_knob_set_per_scheduler_int64_value(const int knob_id, const char *sched_policy_name, int64_t new_value);

/**
   Set float value for per_scheduler scope.
*/
void starpu_perf_knob_set_per_scheduler_float_value(const int knob_id, const char *sched_policy_name, float new_value);

/**
   Set double value for per_scheduler scope.
*/
void starpu_perf_knob_set_per_scheduler_double_value(const int knob_id, const char *sched_policy_name, double new_value);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERF_STEERING_H__ */
