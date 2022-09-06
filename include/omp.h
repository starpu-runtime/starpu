/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_OPENMP_OMP_H__
#define __STARPU_OPENMP_OMP_H__

#include <starpu.h>

#if defined STARPU_OPENMP
typedef starpu_omp_lock_t omp_lock_t;
typedef starpu_omp_nest_lock_t omp_nest_lock_t;

enum omp_sched_value
{
	omp_sched_undefined = 0,
	omp_sched_static    = 1,
	omp_sched_dynamic   = 2,
	omp_sched_guided    = 3,
	omp_sched_auto	    = 4,
	omp_sched_runtime   = 5
};

enum omp_proc_bind_value
{
	omp_proc_bind_undefined = -1,
	omp_proc_bind_false	= 0,
	omp_proc_bind_true	= 1,
	omp_proc_bind_master	= 2,
	omp_proc_bind_close	= 3,
	omp_proc_bind_spread	= 4
};

#ifdef __cplusplus
extern "C" {
#define __STARPU_OMP_NOTHROW throw()
#else
#define __STARPU_OMP_NOTHROW __attribute__((__nothrow__))
#endif

extern void omp_set_num_threads(int threads) __STARPU_OMP_NOTHROW;
extern int omp_get_num_threads() __STARPU_OMP_NOTHROW;
extern int omp_get_thread_num() __STARPU_OMP_NOTHROW;
extern int omp_get_max_threads() __STARPU_OMP_NOTHROW;
extern int omp_get_num_procs(void) __STARPU_OMP_NOTHROW;
extern int omp_in_parallel(void) __STARPU_OMP_NOTHROW;
extern void omp_set_dynamic(int dynamic_threads) __STARPU_OMP_NOTHROW;
extern int omp_get_dynamic(void) __STARPU_OMP_NOTHROW;
extern void omp_set_nested(int nested) __STARPU_OMP_NOTHROW;
extern int omp_get_nested(void) __STARPU_OMP_NOTHROW;
extern int omp_get_cancellation(void) __STARPU_OMP_NOTHROW;
extern void omp_set_schedule(enum omp_sched_value kind, int modifier) __STARPU_OMP_NOTHROW;
extern void omp_get_schedule(enum omp_sched_value *kind, int *modifier) __STARPU_OMP_NOTHROW;
extern int omp_get_thread_limit(void) __STARPU_OMP_NOTHROW;
extern void omp_set_max_active_levels(int max_levels) __STARPU_OMP_NOTHROW;
extern int omp_get_max_active_levels(void) __STARPU_OMP_NOTHROW;
extern int omp_get_level(void) __STARPU_OMP_NOTHROW;
extern int omp_get_ancestor_thread_num(int level) __STARPU_OMP_NOTHROW;
extern int omp_get_team_size(int level) __STARPU_OMP_NOTHROW;
extern int omp_get_active_level(void) __STARPU_OMP_NOTHROW;
extern int omp_in_final(void) __STARPU_OMP_NOTHROW;
extern enum omp_proc_bind_value omp_get_proc_bind(void) __STARPU_OMP_NOTHROW;
extern int omp_get_num_places(void) __STARPU_OMP_NOTHROW;
extern int omp_get_place_num_procs(int place_num) __STARPU_OMP_NOTHROW;
extern void omp_get_place_proc_ids(int place_num, int *ids) __STARPU_OMP_NOTHROW;
extern int omp_get_place_num(void) __STARPU_OMP_NOTHROW;
extern int omp_get_partition_num_places(void) __STARPU_OMP_NOTHROW;
extern void omp_get_partition_place_nums(int *place_nums) __STARPU_OMP_NOTHROW;
extern void omp_set_default_device(int device_num) __STARPU_OMP_NOTHROW;
extern int omp_get_default_device(void) __STARPU_OMP_NOTHROW;
extern int omp_get_num_devices(void) __STARPU_OMP_NOTHROW;
extern int omp_get_num_teams(void) __STARPU_OMP_NOTHROW;
extern int omp_get_team_num(void) __STARPU_OMP_NOTHROW;
extern int omp_is_initial_device(void) __STARPU_OMP_NOTHROW;
extern int omp_get_initial_device(void) __STARPU_OMP_NOTHROW;
extern int omp_get_max_task_priority(void) __STARPU_OMP_NOTHROW;
extern void omp_init_lock(omp_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_destroy_lock(omp_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_set_lock(omp_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_unset_lock(omp_lock_t *lock) __STARPU_OMP_NOTHROW;
extern int omp_test_lock(omp_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_init_nest_lock(omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_destroy_nest_lock(omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_set_nest_lock(omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_unset_nest_lock(omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;
extern int omp_test_nest_lock(omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;
extern void omp_atomic_fallback_inline_begin(void) __STARPU_OMP_NOTHROW;
extern void omp_atomic_fallback_inline_end(void) __STARPU_OMP_NOTHROW;
extern double omp_get_wtime(void) __STARPU_OMP_NOTHROW;
extern double omp_get_wtick(void) __STARPU_OMP_NOTHROW;
extern void *omp_get_local_cuda_stream(void) __STARPU_OMP_NOTHROW;

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_OPENMP && !STARPU_DONT_INCLUDE_OPENMP_HEADERS */
#endif /* __STARPU_OPENMP_OMP_H__ */
