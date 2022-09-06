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

#ifndef __STARPU_PERF_MONITORING_H__
#define __STARPU_PERF_MONITORING_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Perf_Monitoring Performance Monitoring Counters
   @brief This section describes the interface to access performance monitoring counters.
   @{
*/

/**
   @name API
   \anchor PM_API
   @{
*/
/**
   Enum of all possible performance counter scopes.
 */
enum starpu_perf_counter_scope
{
	starpu_perf_counter_scope_undefined   = 0, /**< undefined scope */
	starpu_perf_counter_scope_global      = 2, /**< global scope */
	starpu_perf_counter_scope_per_worker  = 4, /**< per-worker scope */
	starpu_perf_counter_scope_per_codelet = 6  /**< per-codelet scope */
};

/**
  Enum of all possible performance counter value type.
 */
enum starpu_perf_counter_type
{
	starpu_perf_counter_type_undefined = 0, /**< underfined value type */
	starpu_perf_counter_type_int32	   = 1, /**< signed 32-bit integer value */
	starpu_perf_counter_type_int64	   = 2, /**< signed 64-bit integer value */
	starpu_perf_counter_type_float	   = 3, /**< 32-bit single precision floating-point value */
	starpu_perf_counter_type_double	   = 4	/**< 64-bit double precision floating-point value */
};

struct starpu_perf_counter_listener;
struct starpu_perf_counter_sample;
struct starpu_perf_counter_set;

/**
  Start collecting performance counter values.
  */
void starpu_perf_counter_collection_start(void);
/**
  Stop collecting performance counter values.
  */
void starpu_perf_counter_collection_stop(void);

/**
  Translate scope name constant string to scope id.
  */
int starpu_perf_counter_scope_name_to_id(const char *name);
/**
  Translate scope id to scope name constant string.
  */
const char *starpu_perf_counter_scope_id_to_name(enum starpu_perf_counter_scope scope);

/**
  Translate type name constant string to type id.
  */
int starpu_perf_counter_type_name_to_id(const char *name);
/**
  Translate type id to type name constant string.
  */
const char *starpu_perf_counter_type_id_to_name(enum starpu_perf_counter_type type);

/**
  Return the number of performance counters for the given scope.
  */
int starpu_perf_counter_nb(enum starpu_perf_counter_scope scope);
/**
  Translate a performance counter name to its id.
  */
int starpu_perf_counter_name_to_id(enum starpu_perf_counter_scope scope, const char *name);
/**
  Translate a performance counter rank in its scope to its counter id.
  */
int starpu_perf_counter_nth_to_id(enum starpu_perf_counter_scope scope, int nth);
/**
  Translate a counter id to its name constant string.
  */
const char *starpu_perf_counter_id_to_name(int id);
/**
  Return the counter's type id.
  */
int starpu_perf_counter_get_type_id(int id);
/**
  Return the counter's help string.
  */
const char *starpu_perf_counter_get_help_string(int id);

/**
  Display the list of counters defined in the given scope.
  */
void starpu_perf_counter_list_avail(enum starpu_perf_counter_scope scope);
/**
  Display the list of counters defined in all scopes.
  */
void starpu_perf_counter_list_all_avail(void);

/**
  Allocate a new performance counter set.
  */
struct starpu_perf_counter_set *starpu_perf_counter_set_alloc(enum starpu_perf_counter_scope scope);
/**
  Free a performance counter set.
  */
void starpu_perf_counter_set_free(struct starpu_perf_counter_set *set);

/**
  Enable a given counter in the set.
  */
void starpu_perf_counter_set_enable_id(struct starpu_perf_counter_set *set, int id);
/**
  Disable a given counter in the set.
  */
void starpu_perf_counter_set_disable_id(struct starpu_perf_counter_set *set, int id);

/**
  Initialize a new performance counter listener.
  */
struct starpu_perf_counter_listener *starpu_perf_counter_listener_init(struct starpu_perf_counter_set *set, void (*callback)(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context), void *user_arg);
/**
  End a performance counter listener.
  */
void starpu_perf_counter_listener_exit(struct starpu_perf_counter_listener *listener);

/**
  Set a listener for the global scope.
  */
void starpu_perf_counter_set_global_listener(struct starpu_perf_counter_listener *listener);
/**
  Set a listener for the per_worker scope on a given worker.
  */
void starpu_perf_counter_set_per_worker_listener(unsigned workerid, struct starpu_perf_counter_listener *listener);
/**
  Set a common listener for all workers.
  */
void starpu_perf_counter_set_all_per_worker_listeners(struct starpu_perf_counter_listener *listener);
/**
  Set a per_codelet listener for a codelet.
  */
void starpu_perf_counter_set_per_codelet_listener(struct starpu_codelet *cl, struct starpu_perf_counter_listener *listener);

/**
  Unset the global listener.
  */
void starpu_perf_counter_unset_global_listener(void);
/**
  Unset the per_worker listener.
  */
void starpu_perf_counter_unset_per_worker_listener(unsigned workerid);
/**
  Unset all per_worker listeners.
  */
void starpu_perf_counter_unset_all_per_worker_listeners(void);
/**
  Unset a per_codelet listener.
  */
void starpu_perf_counter_unset_per_codelet_listener(struct starpu_codelet *cl);

/**
  Read an int32 counter value from a sample.
  */
int32_t starpu_perf_counter_sample_get_int32_value(struct starpu_perf_counter_sample *sample, const int counter_id);
/**
  Read an int64 counter value from a sample.
  */
int64_t starpu_perf_counter_sample_get_int64_value(struct starpu_perf_counter_sample *sample, const int counter_id);
/**
  Read a float counter value from a sample.
  */
float starpu_perf_counter_sample_get_float_value(struct starpu_perf_counter_sample *sample, const int counter_id);
/**
  Read a double counter value from a sample.
  */
double starpu_perf_counter_sample_get_double_value(struct starpu_perf_counter_sample *sample, const int counter_id);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERF_MONITORING_H__ */
