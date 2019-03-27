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

#ifndef __STARPU_PERF_MONITORING_H__
#define __STARPU_PERF_MONITORING_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Perf_Monitoring Perf_Monitoring
   @{
*/

enum starpu_perf_counter_scope
{
	starpu_perf_counter_scope_undefined     = 0,
	starpu_perf_counter_scope_global        = 2,
	starpu_perf_counter_scope_per_worker    = 4,
	starpu_perf_counter_scope_per_codelet   = 6
};

enum starpu_perf_counter_type
{
	starpu_perf_counter_type_undefined = 0,
	starpu_perf_counter_type_int32     = 1,
	starpu_perf_counter_type_int64     = 2,
	starpu_perf_counter_type_float     = 3,
	starpu_perf_counter_type_double    = 4
};

struct starpu_perf_counter_listener;
struct starpu_perf_counter_sample;
struct starpu_perf_counter_set;

int starpu_perf_counter_scope_name_to_id(const char *name);
const char *starpu_perf_counter_scope_id_to_name(enum starpu_perf_counter_scope scope);

int starpu_perf_counter_type_name_to_id(const char *name);
const char *starpu_perf_counter_type_id_to_name(enum starpu_perf_counter_type type);

int starpu_perf_counter_nb(enum starpu_perf_counter_scope scope);
int starpu_perf_counter_name_to_id(enum starpu_perf_counter_scope scope, const char *name);
int starpu_perf_counter_nth_to_id(enum starpu_perf_counter_scope scope, int nth);
const char *starpu_perf_counter_id_to_name(int id);
int starpu_perf_counter_get_type_id(int id);
const char *starpu_perf_counter_get_help_string(int id);

void starpu_perf_counter_list_avail(enum starpu_perf_counter_scope scope);
void starpu_perf_counter_list_all_avail(enum starpu_perf_counter_scope scope);

struct starpu_perf_counter_set *starpu_perf_counter_set_alloc(enum starpu_perf_counter_scope scope);
void starpu_perf_counter_set_free(struct starpu_perf_counter_set *set);

void starpu_perf_counter_set_enable_id(struct starpu_perf_counter_set *set, int id);
void starpu_perf_counter_set_disable_id(struct starpu_perf_counter_set *set, int id);

struct starpu_perf_counter_listener *starpu_perf_counter_listener_init(struct starpu_perf_counter_set *set, void (*callback)(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context), void *user_arg);
void starpu_perf_counter_listener_exit(struct starpu_perf_counter_listener *listener);

void starpu_perf_counter_set_global_listener(struct starpu_perf_counter_listener *listener);
void starpu_perf_counter_set_per_worker_listener(unsigned workerid, struct starpu_perf_counter_listener *listener);
void starpu_perf_counter_set_all_per_worker_listeners(struct starpu_perf_counter_listener *listener);
void starpu_perf_counter_set_per_codelet_listener(struct starpu_codelet *cl, struct starpu_perf_counter_listener *listener);

void starpu_perf_counter_unset_global_listener();
void starpu_perf_counter_unset_per_worker_listener(unsigned workerid);
void starpu_perf_counter_unset_all_per_worker_listeners(void);
void starpu_perf_counter_unset_per_codelet_listener(struct starpu_codelet *cl);

int32_t starpu_perf_counter_sample_get_int32_value(struct starpu_perf_counter_sample *sample, const int counter_id);
int64_t starpu_perf_counter_sample_get_int64_value(struct starpu_perf_counter_sample *sample, const int counter_id);
float starpu_perf_counter_sample_get_float_value(struct starpu_perf_counter_sample *sample, const int counter_id);
double starpu_perf_counter_sample_get_double_value(struct starpu_perf_counter_sample *sample, const int counter_id);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERF_MONITORING_H__ */
