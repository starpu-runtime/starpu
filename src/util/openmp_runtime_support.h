/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __OPENMP_RUNTIME_SUPPORT_H__
#define __OPENMP_RUNTIME_SUPPORT_H__

/** @file */

#include <starpu.h>

#ifdef STARPU_OPENMP
#include <common/list.h>
#include <common/starpu_spinlock.h>
#include <common/uthash.h>

/** ucontexts have been deprecated as of POSIX 1-2004
 * _XOPEN_SOURCE required at least on OS/X
 *
 * TODO: add detection in configure.ac
 */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE
#endif
#include <ucontext.h>

extern starpu_pthread_key_t omp_thread_key;
extern starpu_pthread_key_t omp_task_key;

/**
 * Arbitrary limit on the number of nested parallel sections
 */
#define STARPU_OMP_MAX_ACTIVE_LEVELS 1

/**
 * Possible abstract names for OpenMP places
 */
enum starpu_omp_place_name
{
	starpu_omp_place_undefined = 0,
	starpu_omp_place_threads   = 1,
	starpu_omp_place_cores     = 2,
	starpu_omp_place_sockets   = 3,
	starpu_omp_place_numerical = 4 /** place specified numerically */
};

struct starpu_omp_numeric_place
{
	int excluded_place;
	int *included_numeric_items;
	int nb_included_numeric_items;
	int *excluded_numeric_items;
	int nb_excluded_numeric_items;
};

/**
 * OpenMP place for thread afinity, defined by the OpenMP spec
 */
struct starpu_omp_place
{
	int abstract_name;
	int abstract_excluded;
	int abstract_length;
	struct starpu_omp_numeric_place *numeric_places;
	int nb_numeric_places;
};

/**
 * Internal Control Variables (ICVs) declared following
 * OpenMP 4.0.0 spec section 2.3.1
 */
struct starpu_omp_data_environment_icvs
{
	/** parallel region icvs */
	int dyn_var;
	int nest_var;
	int *nthreads_var; /** nthreads_var ICV is a list */
	int thread_limit_var;

	int active_levels_var;
	int levels_var;
	int *bind_var; /** bind_var ICV is a list */

	/** loop region icvs */
	int run_sched_var;
	unsigned long long run_sched_chunk_var;

	/** program execution icvs */
	int default_device_var;
	int max_task_priority_var;
};

struct starpu_omp_device_icvs
{
	/** parallel region icvs */
	int max_active_levels_var;

	/** loop region icvs */
	int def_sched_var;
	unsigned long long def_sched_chunk_var;

	/** program execution icvs */
	int stacksize_var;
	int wait_policy_var;
};

struct starpu_omp_implicit_task_icvs
{
	/** parallel region icvs */
	int place_partition_var;
};

struct starpu_omp_global_icvs
{
	/** program execution icvs */
	int cancel_var;
};

struct starpu_omp_initial_icv_values
{
	int dyn_var;
	int nest_var;
	int *nthreads_var;
	int run_sched_var;
	unsigned long long run_sched_chunk_var;
	int def_sched_var;
	unsigned long long def_sched_chunk_var;
	int *bind_var;
	int stacksize_var;
	int wait_policy_var;
	int thread_limit_var;
	int max_active_levels_var;
	int active_levels_var;
	int levels_var;
	int place_partition_var;
	int cancel_var;
	int default_device_var;
	int max_task_priority_var;

	/** not a real ICV, but needed to store the contents of OMP_PLACES */
	struct starpu_omp_place places;
};

struct starpu_omp_task_group
{
	int descendent_task_count;
	struct starpu_omp_task *leader_task;
	struct starpu_omp_task_group *p_previous_task_group;
};

struct starpu_omp_task_link
{
	struct starpu_omp_task *task;
	struct starpu_omp_task_link *next;
};

struct starpu_omp_condition
{
	struct starpu_omp_task_link *contention_list_head;
};

struct starpu_omp_critical
{
	UT_hash_handle hh;
	struct _starpu_spinlock lock;
	unsigned state;
	struct starpu_omp_task_link *contention_list_head;
	const char *name;
};

enum starpu_omp_task_state
{
	starpu_omp_task_state_clear      = 0,
	starpu_omp_task_state_preempted  = 1,
	starpu_omp_task_state_terminated = 2,
	starpu_omp_task_state_zombie     = 3,

	/** target tasks are non-preemptible tasks, without dedicated stack and OpenMP Runtime Support context */
	starpu_omp_task_state_target     = 4,
};

enum starpu_omp_task_wait_on
{
	starpu_omp_task_wait_on_task_childs  = 1 << 0,
	starpu_omp_task_wait_on_region_tasks = 1 << 1,
	starpu_omp_task_wait_on_barrier      = 1 << 2,
	starpu_omp_task_wait_on_group        = 1 << 3,
	starpu_omp_task_wait_on_critical     = 1 << 4,
	starpu_omp_task_wait_on_ordered      = 1 << 5,
	starpu_omp_task_wait_on_lock         = 1 << 6,
	starpu_omp_task_wait_on_nest_lock    = 1 << 7,
};

enum starpu_omp_task_flags
{
	STARPU_OMP_TASK_FLAGS_IMPLICIT	     = 1 << 0,
	STARPU_OMP_TASK_FLAGS_UNDEFERRED     = 1 << 1,
	STARPU_OMP_TASK_FLAGS_FINAL	     = 1 << 2,
	STARPU_OMP_TASK_FLAGS_UNTIED	     = 1 << 3,
};

LIST_TYPE(starpu_omp_task,
	struct starpu_omp_implicit_task_icvs icvs;
	struct starpu_omp_task *parent_task;
	struct starpu_omp_thread *owner_thread;
	struct starpu_omp_region *owner_region;
	struct starpu_omp_region *nested_region;
	int rank;
	int child_task_count;
	struct starpu_omp_task_group *task_group;
	struct _starpu_spinlock lock;
	int transaction_pending;
	int wait_on;
	int barrier_count;
	int single_id;
	int single_first;
	int loop_id;
	unsigned long long ordered_first_i;
	unsigned long long ordered_nb_i;
	int sections_id;
	struct starpu_omp_data_environment_icvs data_env_icvs;
	struct starpu_omp_implicit_task_icvs implicit_task_icvs;
	struct handle_entry *registered_handles;

	struct starpu_task *starpu_task;
	struct starpu_codelet cl;
	void **starpu_buffers;
	void *starpu_cl_arg;

	/** actual task function to be run */
	void (*cpu_f)(void **starpu_buffers, void *starpu_cl_arg);
#ifdef STARPU_USE_CUDA
	void (*cuda_f)(void **starpu_buffers, void *starpu_cl_arg);
#endif
#ifdef STARPU_USE_OPENCL
	void (*opencl_f)(void **starpu_buffers, void *starpu_cl_arg);
#endif

	enum starpu_omp_task_state state;
	enum starpu_omp_task_flags flags;

	/*
	 * context to store the processing state of the task
	 * in case of blocking/recursive task operation
	 */
	ucontext_t ctx;

	/*
	 * stack to execute the task over, to be able to switch
	 * in case blocking/recursive task operation
	 */
	void *stack;

	/*
	 * Valgrind stack id
	 */
	int stack_vg_id;

	size_t stacksize;

   /*
    * taskloop attribute
    * */
   int is_loop;
   unsigned long long nb_iterations;
   unsigned long long grainsize;
   unsigned long long chunk;
   unsigned long long begin_i;
   unsigned long long end_i;
)

LIST_TYPE(starpu_omp_thread,

	UT_hash_handle hh;
	struct starpu_omp_task *current_task;
	struct starpu_omp_region *owner_region;

	/*
	 * stack to execute the initial thread over
	 * when preempting the initial task
	 * note: should not be used for other threads
	 */
	void *initial_thread_stack;
	/*
	 * Valgrind stack id
	 */
	int initial_thread_stack_vg_id;

	/*
	 * context to store the 'scheduler' state of the thread,
	 * to which the execution of thread comes back upon a
	 * blocking/recursive task operation
	 */
	ucontext_t ctx;

	struct starpu_driver starpu_driver;
	struct _starpu_worker *worker;
)

struct _starpu_omp_lock_internal
{
	struct _starpu_spinlock lock;
	struct starpu_omp_condition cond;
	unsigned state;
};

struct _starpu_omp_nest_lock_internal
{
	struct _starpu_spinlock lock;
	struct starpu_omp_condition cond;
	unsigned state;
	struct starpu_omp_task *owner_task;
	unsigned nesting;
};

struct starpu_omp_loop
{
	int id;
	unsigned long long next_iteration;
	int nb_completed_threads;
	struct starpu_omp_loop *next_loop;
	struct _starpu_spinlock ordered_lock;
	struct starpu_omp_condition ordered_cond;
	unsigned long long ordered_iteration;
};

struct starpu_omp_sections
{
	int id;
	unsigned long long next_section_num;
	int nb_completed_threads;
	struct starpu_omp_sections *next_sections;
};

struct starpu_omp_region
{
	struct starpu_omp_data_environment_icvs icvs;
	struct starpu_omp_region *parent_region;
	struct starpu_omp_device *owner_device;
	struct starpu_omp_thread *master_thread;
	/** note: the list of threads does not include the master_thread */
	struct starpu_omp_thread_list thread_list;
	/** list of implicit omp tasks created to run the region */
	struct starpu_omp_task **implicit_task_array;
	/** include both the master thread and the region own threads */
	int nb_threads;
	struct _starpu_spinlock lock;
	struct starpu_omp_task *waiting_task;
	int barrier_count;
	int bound_explicit_task_count;
	int single_id;
	void *copy_private_data;
	int level;
	struct starpu_omp_loop *loop_list;
	struct starpu_omp_sections *sections_list;
	struct starpu_task *continuation_starpu_task;
	struct handle_entry *registered_handles;
	struct _starpu_spinlock registered_handles_lock;
};

struct starpu_omp_device
{
	struct starpu_omp_device_icvs icvs;

	/** atomic fallback implementation lock */
	struct _starpu_spinlock atomic_lock;
};

struct starpu_omp_global
{
	struct starpu_omp_global_icvs icvs;
	struct starpu_omp_task *initial_task;
	struct starpu_omp_thread *initial_thread;
	struct starpu_omp_region *initial_region;
	struct starpu_omp_device *initial_device;
	struct starpu_omp_critical *default_critical;
	struct starpu_omp_critical *named_criticals;
	struct _starpu_spinlock named_criticals_lock;
	struct starpu_omp_thread *hash_workers;
	struct _starpu_spinlock hash_workers_lock;
	struct starpu_arbiter *default_arbiter;
	unsigned nb_starpu_cpu_workers;
	int *starpu_cpu_worker_ids;
	int environment_valid;
};

/*
 * internal global variables
 */
extern struct starpu_omp_initial_icv_values *_starpu_omp_initial_icv_values;
extern struct starpu_omp_global *_starpu_omp_global_state;
extern double _starpu_omp_clock_ref;

/*
 * internal API
 */
void _starpu_omp_environment_init(void);
void _starpu_omp_environment_exit(void);
int _starpu_omp_environment_check(void);
struct starpu_omp_thread *_starpu_omp_get_thread(void);
struct starpu_omp_region *_starpu_omp_get_region_at_level(int level);
struct starpu_omp_task *_starpu_omp_get_task(void);
int _starpu_omp_get_region_thread_num(const struct starpu_omp_region *const region);
void _starpu_omp_dummy_init(void);
void _starpu_omp_dummy_shutdown(void);
#endif // STARPU_OPENMP

#endif // __OPENMP_RUNTIME_SUPPORT_H__
