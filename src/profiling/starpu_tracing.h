/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria, Télécom SudParis
 * Copyright (C) 2023-2025  École de Technologie Supérieure (ETS, Montréal)
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

#ifndef STARPU_TRACE_H
#define STARPU_TRACE_H

#ifdef STARPU_PROF_TOOL
#include "callbacks/callbacks.h"
#endif

#ifdef STARPU_PROF_TASKSTUBS

#include <tasktimer.h>
#include <tool_api.h>

uint64_t new_guid();
#endif

#define STARPU_TRACE_API_VERSION 1

struct _starpu_tag;
struct _starpu_data_request;
struct _starpu_worker;
struct _starpu_job;
struct starpu_sched_component;

/* Initialize any existing tracing/profiling tool  */
int _starpu_trace_initialize();
/* Finalize any existing tracing/profiling tool  */
int _starpu_trace_finalize();

/* Set profiling status */
int _starpu_trace_set_profiling(int status);
/* Stop PAPI counters */
int _starpu_trace_papi_task_event(int event_id, struct starpu_task* task, long long int value);
/* Called with "start_profiling" at the beginning of FXT profiling, with "stop_profiling" at the end of FXT profiling, and with an arbitrary string by FXT (FXT only) */
int _starpu_trace_meta(const char* S);
/* FXT user event (FXT only) */
int _starpu_trace_user_event(unsigned long code);
/* Unused */
int _starpu_trace_event_always(const char* S);
/* FXT event (FXT only) */
int _starpu_trace_event(const char* S);
/* Unused */
int _starpu_trace_event_verbose(const char* S);
/* Unused */
int _starpu_trace_thread_event(const char* S);

/* Register a new memory node */
int _starpu_trace_new_mem_node(int nodeid);
/* Beginning of the initialization of the driver for a worker */
int _starpu_trace_worker_init_start(struct _starpu_worker *worker, enum starpu_worker_archtype archtype, unsigned sync);
/* End of the initialization of the driver for a worker */
int _starpu_trace_worker_init_end(struct _starpu_worker *worker, enum starpu_worker_archtype archtype);
/* When a new worker starts, register it */
int _starpu_trace_register_thread(int bindid);
/* Initialize a worker at the beginning of the execution of the application */
int _starpu_trace_worker_initialize();
/* Finalize a worker at the end of the execution of the application */
int _starpu_trace_worker_finalize();
/* Beginning of the finalization of a worker */
int _starpu_trace_worker_deinit_start();
/* End of the finalization of a worker */
int _starpu_trace_worker_deinit_end(unsigned workerid, enum starpu_worker_archtype workerkind);

/* Start the execution of a codelet */
int _starpu_trace_start_codelet_body(struct _starpu_job *job, int nimpl, struct starpu_perfmodel_arch* perf_arch, int workerid, int rank);
/* End the execution of a codelet */
int _starpu_trace_end_codelet_body(struct _starpu_job *job, unsigned nimpl, struct starpu_perfmodel_arch* perf_arch, int workerid, int rank);
/* Start the execution of a codelet on the worker it was assigned to */
int _starpu_trace_start_executing(struct _starpu_job *j, struct starpu_task *worker_task, struct _starpu_worker *cpu_args, void* func);
/* End the execution of the codelet on the worker */
int _starpu_trace_end_executing(struct _starpu_job *job, struct _starpu_worker *worker);
/* Before a call to the epilogue callback */
int _starpu_trace_start_callback(struct _starpu_job *job);
/* After a call to the epilogue callback */
int _starpu_trace_end_callback(struct _starpu_job *job);
/* Push a task on a specific worker, called once per worker */
int _starpu_trace_job_push(struct starpu_task* task, int prio);
/* Pop a task from the scheduler, either to resubmit it or at the end of its execution */
int _starpu_trace_job_pop(struct starpu_task* task, int prio);
/* Set a task counter */
int _starpu_trace_update_task_cnt(int counter);

/* Begin fetching a task's data input */
int _starpu_trace_start_fetch_input(struct _starpu_job *job);
/* End fetching a task's data input */
int _starpu_trace_end_fetch_input(struct _starpu_job *job);
/* Begin pushing a task output */
int _starpu_trace_start_push_output(struct _starpu_job *job);
/* End pushing a task output (unused) */
int _starpu_trace_worker_end_fetch_input(struct _starpu_job *job, int id);
/* Beginning of a data fetch operation for a given task */
int _starpu_trace_worker_start_fetch_input(struct _starpu_job *job, int id);
/* End of a data fetch operation for a given task */
int _starpu_trace_end_push_output(struct _starpu_job *job);

/* Declare a tag */
int _starpu_trace_tag(starpu_tag_t* tag, struct _starpu_job *job);
/* Declare a tag's dependencies (called once per dependency) */
int _starpu_trace_tag_deps(starpu_tag_t* tag_child, starpu_tag_t* tag_parent);
/* Declare a tasks's dependencies (called once per dependency) */
int _starpu_trace_task_deps(struct _starpu_job *job_prev, struct _starpu_job *job_succ);
/* Release task dependency and terminate the job */
int _starpu_trace_task_end_dep(struct _starpu_job *job_prev, struct _starpu_job *job_succ);
/* Add a ghost dependency */
int _starpu_trace_ghost_task_deps(unsigned ghost_prev_id, struct _starpu_job *job_succ);
/* Push a non-root recursive task */
int _starpu_trace_recursive_task_deps(unsigned long prev_id, struct _starpu_job *job_succ);

/* Execute a recursive task */
int _starpu_trace_recursive_task(struct _starpu_job *job);
/* Exclude a task from the DAG */
int _starpu_trace_task_exclude_from_dag(struct _starpu_job *job);
/* Set the task's line number as set by the programmer to be used by an external profiling system. */
int _starpu_trace_task_line(struct _starpu_job *job);
/* Set the task's name as set by the programmer to be used by an external profiling system. */
int _starpu_trace_task_name(struct _starpu_job *job);
/* Set the task's color as set by the programmer to be used by an external profiling system. */
int _starpu_trace_task_color(struct _starpu_job *job);
/* Set the task's name, line number, and color as set by the programmer to be used by an external profiling system. */
int _starpu_trace_task_name_line_color(struct _starpu_job *job);
/* The task execution is finished, it is going to be destroyed */
int _starpu_trace_task_done(struct _starpu_job *job);
/* Notify that a tag is done */
int _starpu_trace_tag_done(struct _starpu_tag* tag);

/* Set the data's name */
int _starpu_trace_data_name(starpu_data_handle_t *handle, const char* name);
/* Set the data's coordinates array */
int _starpu_trace_data_coordinates(starpu_data_handle_t *handle, unsigned dim, int v[]);
/* Copy data */
int _starpu_trace_data_copy(unsigned src_node, unsigned dst_node, size_t size);
/* Set all my children's handles as not being used in the future */
int _starpu_trace_data_wont_use(starpu_data_handle_t *handle);
/* Set all my memory chunks as not being used in the future */
int _starpu_trace_data_doing_wont_use(starpu_data_handle_t *handle);
/* Start data copy request */
int _starpu_trace_start_driver_copy(unsigned src_node, unsigned dst_node, size_t size, unsigned long com_id, enum starpu_is_prefetch prefetch, starpu_data_handle_t *handle);
/* Data copy request completed */
int _starpu_trace_end_driver_copy(unsigned src_node, unsigned dst_node, size_t size, unsigned long com_id,enum starpu_is_prefetch prefetch);
/* Start asynchronous data request */
int _starpu_trace_start_driver_copy_async(unsigned src_node, unsigned dst_node);
/* Asynchronous data request completed */
int _starpu_trace_end_driver_copy_async(unsigned src_node, unsigned dst_node);

/* Register a data handle */
int _starpu_trace_handle_data_register(starpu_data_handle_t *handle);
/* Unregister a data handle */
int _starpu_trace_handle_data_unregister(starpu_data_handle_t *handle);
/* Set the data's state as invalid */
int _starpu_trace_data_state_invalid(starpu_data_handle_t *handle, unsigned node);
/* Set the data's owner */
int _starpu_trace_data_state_owner(starpu_data_handle_t *handle, unsigned node);
/* Set the data's state as shared */
int _starpu_trace_data_state_shared(starpu_data_handle_t *handle, unsigned node);
/* Create a data request */
int _starpu_trace_data_request_created(starpu_data_handle_t *handle, int orig, int dest, int prio, enum starpu_is_prefetch is_prefetch, struct _starpu_data_request *req);

/* Start unapplying a filter */
int _starpu_trace_start_unpartition(starpu_data_handle_t *handle, unsigned memnode);
/* Finished unapplying a filter */
int _starpu_trace_end_unpartition(starpu_data_handle_t *handle, unsigned memnode);

/* Schedule a task (work stealing scheduling policy) */
int _starpu_trace_work_stealing(unsigned empty_q, unsigned victim_q);
/* Scheduling start, so set the status as "scheduling" */
int _starpu_trace_worker_scheduling_start();
/* Scheduling done, so set the status as "scheduling done" */
int _starpu_trace_worker_scheduling_end();
/* Enqueue a task into the list of tasks explicitly attached to a worker */
int _starpu_trace_worker_scheduling_push();
/* After the scheduler has pushed a task to a queue but just before releasing mutexes */
int _starpu_trace_worker_scheduling_pop();
/* Set status as "sleeping" */
int _starpu_trace_worker_sleep_start();
/* Wake up, so clear the status */
int _starpu_trace_worker_sleep_end();
/* Submit a task */
int _starpu_trace_task_submit(struct _starpu_job *job, long iter, long subiter);
/* Before a task is submitted to the scheduler */
int _starpu_trace_task_submit_start();
/* After a task is submitted to the scheduler */
int _starpu_trace_task_submit_end();
/* Throttle a task to wait until the number of submitted tasks gets below a certain limit */
int _starpu_trace_task_throttle_start();
/* Un-throttle a task */
int _starpu_trace_task_throttle_end();
/* Before the creation of the data structure that holds a task */
int _starpu_trace_task_build_start();
/* After the creation of the data structure that holds a task */
int _starpu_trace_task_build_end();
/* Wait until a task is started */
int _starpu_trace_task_wait_start(struct _starpu_job *job);
/* After a task is started */
int _starpu_trace_task_wait_end();
/* Before waiting for all the tasks of the scheduling context */
int _starpu_trace_task_wait_for_all_start();
/* After waiting for all the tasks of the scheduling context */
int _starpu_trace_task_wait_for_all_end();
/* Unused, see _starpu_trace_task_wait_for_all_start() and  _starpu_trace_task_wait_for_all_end() */
int _starpu_trace_task_wait_for_all();

/* Push a task (prio scheduler) */
int _starpu_trace_sched_component_push_prio(struct starpu_sched_component * component, unsigned ntasks, double exp_len);
/* Pop a task (prio scheduler) */
int _starpu_trace_sched_component_pop_prio(struct starpu_sched_component * component, unsigned ntasks, double exp_len);
/* Create a new scheduling component */
int _starpu_trace_sched_component_new(struct starpu_sched_component* component);
/* Attach a component to its parent (scheduling) */
int _starpu_trace_sched_component_connect(struct starpu_sched_component* parent, struct starpu_sched_component* child);
/* Push a task to a component (scheduling) */
int _starpu_trace_sched_component_push(struct starpu_sched_component* from, struct starpu_sched_component* to, struct starpu_task* task, int prio);
/* Pull a task from a component (scheduling) */
int _starpu_trace_sched_component_pull(struct starpu_sched_component* from, struct starpu_sched_component* to, struct starpu_task* task);

/* Before sending a notification to the scheduling context (only if STARPU_USE_SC_HYPERVISOR is enabled ) */
int _starpu_trace_hypervisor_begin();
/* After sending a notification to the scheduling context (only if STARPU_USE_SC_HYPERVISOR is enabled ) */
int _starpu_trace_hypervisor_end();

/* Beginning of a memory allocation */
int _starpu_trace_start_alloc(unsigned memnode, size_t size, starpu_data_handle_t *handle, enum starpu_is_prefetch is_prefetch);
/* End of a memory allocation */
int _starpu_trace_end_alloc(unsigned memnode, starpu_data_handle_t *handle, starpu_ssize_t r);
/* Beginning of a memory allocation using allocation cache */
int _starpu_trace_start_alloc_reuse(unsigned memnode, size_t size, starpu_data_handle_t *handle, enum starpu_is_prefetch is_prefetch);
/* End of a memory allocation using allocation cache */
int _starpu_trace_end_alloc_reuse(unsigned memnode, starpu_data_handle_t *handle, starpu_ssize_t r);
/* Before memory is freeed */
int _starpu_trace_start_free(unsigned memnode, size_t size, starpu_data_handle_t *handle);
/* After memory is freeed */
int _starpu_trace_end_free(unsigned memnode, starpu_data_handle_t *handle);
/* Before a subtree is transfered to a node */
int _starpu_trace_start_writeback(unsigned memnode, starpu_data_handle_t *handle);
/* After a subtree is transfered to a node */
int _starpu_trace_end_writeback(unsigned memnode, starpu_data_handle_t *handle);
/* Allocate memory */
int _starpu_trace_used_mem(unsigned memnode, size_t used);
/* Before trying to free the buffers currently in use on the memory node */
int _starpu_trace_start_memreclaim(unsigned memnode,enum starpu_is_prefetch is_prefetch);
/* After trying to free the buffers currently in use on the memory node */
int _starpu_trace_end_memreclaim(unsigned memnode, enum starpu_is_prefetch is_prefetch);
/* Periodic tidy of available memory: start cleaning the memory */
int _starpu_trace_start_writeback_async(unsigned memnode);
/* Periodic tidy of available memory: finished cleaning the memory */
int _starpu_trace_end_writeback_async(unsigned memnode);
/* Memory allocation failed */
int _starpu_trace_memory_full(size_t size);

/* Start a data transfer */
int _starpu_trace_start_transfer(unsigned memnode, struct _starpu_worker *worker);
/* End a data transfer */
int _starpu_trace_end_transfer(unsigned memnode, struct _starpu_worker *worker);
/* Start a progress operation on a data transfer */
int _starpu_trace_start_progress(unsigned memnode, struct _starpu_worker *worker);
/* End a progress operation on a data transfer */
int _starpu_trace_end_progress(unsigned memnode, struct _starpu_worker *worker);

/* Beginning of the function that finds out whether we are to execute the data because we own the data to be written to (MPI mode). */
int _starpu_trace_task_mpi_decode_start();
/* End of the function that finds out whether we are to execute the data because we own the data to be written to (MPI mode). */
int _starpu_trace_task_mpi_decode_end();
/* Start building the necessary data to execute a task, involving a communication to send and receive the necessary data (MPI mode) */
int _starpu_trace_task_mpi_pre_start();
/* Finish building the necessary data to execute a task, involving a communication to send and receive the necessary data (MPI mode) */
int _starpu_trace_task_mpi_pre_end();
/* Start exchanging and clearing data after the execution of a task (MPI mode) */
int _starpu_trace_task_mpi_post_start();
/* Start exchanging and clearing data after the execution of a task (MPI mode) */
int _starpu_trace_task_mpi_post_end();

/* Start locking a pthread mutex */
int _starpu_trace_locking_mutex();
/* A pthread mutex has been locked */
int _starpu_trace_mutex_locked();
/* Start unlocking a pthread mutex */
int _starpu_trace_unlocking_mutex();
/* A pthread mutex has been unlocked */
int _starpu_trace_mutex_unlocked();
/* Start trylock a pthread mutex */
int _starpu_trace_trylock_mutex();
/* Before locking the rw lock */
int _starpu_trace_rdlocking_rwlock();
/* After rw lock has been successfully locked by trylock */
int _starpu_trace_rwlock_rdlocked();
/* Before locking the rw lock */
int _starpu_trace_wrlocking_rwlock();
/* After rw lock has been locked */
int _starpu_trace_rwlock_wrlocked();
/* Before unlocking the rw lock */
int _starpu_trace_unlocking_rwlock();
/* After unlocking the rw lock */
int _starpu_trace_rwlock_unlocked();
/* Unused */
int _starpu_trace_spinlock_conditition();
/* After a lock is taken and the function that called it last is set */
int _starpu_trace_spinlock_locked(const char* file, int line);
/* Before a lock is taken and the function that called it last is set */
int _starpu_trace_locking_spinlock(const char* file, int line);
/* Before a lock is released and the function that called it last is set */
int _starpu_trace_unlocking_spinlock(const char* file, int line);
/* After a lock is released and the function that called it last is set */
int _starpu_trace_spinlock_unlocked(const char* file, int line);
/* Before we try to take a lock with trylock and, if the lock is granted, the function that called it last is set */
int _starpu_trace_trylock_spinlock(const char* file, int line);
/* Before a condition variable is initialized and waited on */
int _starpu_trace_cond_wait_begin();
/* After the wait on the condition wait has returned */
int _starpu_trace_cond_wait_end();
/* Before a barrier */
int _starpu_trace_barrier_wait_begin();
/* After a barrier */
int _starpu_trace_barrier_wait_end();
/* Finished filling the codelet's interfaces */
int _starpu_trace_data_load(int workerid, size_t size);
/* Before a barrier synchronizing the threads */
int _starpu_trace_start_parallel_sync(struct _starpu_job *job);
/* After a barrier synchronizing the threads */
int _starpu_trace_end_parallel_sync(struct _starpu_job *job);


#endif	/* #ifndef STARPU_TRACE_H */
