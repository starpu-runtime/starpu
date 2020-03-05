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

#ifndef __STARPU_OPENMP_H__
#define __STARPU_OPENMP_H__

#include <starpu_config.h>

/**
   @defgroup API_OpenMP_Runtime_Support OpenMP Runtime Support
   @brief This section describes the interface provided for implementing OpenMP runtimes on top of StarPU.
   @{
*/

#if defined STARPU_OPENMP
/**
   Opaque Simple Lock object (\anchor SimpleLock) for inter-task
   synchronization operations.
   \sa starpu_omp_init_lock()
   \sa starpu_omp_destroy_lock()
   \sa starpu_omp_set_lock()
   \sa starpu_omp_unset_lock()
   \sa starpu_omp_test_lock()
*/
typedef struct { void *internal; /**< opaque pointer for internal use */ } starpu_omp_lock_t;

/**
   Opaque Nestable Lock object (\anchor NestableLock) for inter-task
   synchronization operations.
   \sa starpu_omp_init_nest_lock()
   \sa starpu_omp_destroy_nest_lock()
   \sa starpu_omp_set_nest_lock()
   \sa starpu_omp_unset_nest_lock()
   \sa starpu_omp_test_nest_lock()
*/
typedef struct { void *internal; /**< opaque pointer for internal use */  } starpu_omp_nest_lock_t;

/**
   Set of constants for selecting the for loop iteration scheduling
   algorithm (\anchor OMPFor) as defined by the OpenMP specification.
   \sa starpu_omp_for()
   \sa starpu_omp_for_inline_first()
   \sa starpu_omp_for_inline_next()
   \sa starpu_omp_for_alt()
   \sa starpu_omp_for_inline_first_alt()
   \sa starpu_omp_for_inline_next_alt()
*/
enum starpu_omp_sched_value
{
	starpu_omp_sched_undefined = 0, /**< Undefined iteration scheduling algorithm. */
	starpu_omp_sched_static    = 1, /**< \b Static iteration scheduling algorithm.*/
	starpu_omp_sched_dynamic   = 2, /**< \b Dynamic iteration scheduling algorithm.*/
	starpu_omp_sched_guided    = 3, /**< \b Guided iteration scheduling algorithm.*/
	starpu_omp_sched_auto      = 4, /**< \b Automatically choosen iteration scheduling algorithm.*/
	starpu_omp_sched_runtime   = 5 /**< Choice of iteration scheduling algorithm deferred at \b runtime.*/
};

/**
   Set of constants for selecting the processor binding method, as
   defined in the OpenMP specification.
   \sa starpu_omp_get_proc_bind()
*/
enum starpu_omp_proc_bind_value
{
	starpu_omp_proc_bind_undefined  = -1, /**< Undefined processor binding method.*/
	starpu_omp_proc_bind_false  = 0,      /**< Team threads may be moved between places at any time.*/
	starpu_omp_proc_bind_true   = 1,      /**< Team threads may not be moved between places.*/
	starpu_omp_proc_bind_master = 2,      /**< Assign every thread in the team to the same place as the \b master thread.*/
	starpu_omp_proc_bind_close  = 3,      /**< Assign every thread in the team to a place \b close to the parent thread.*/
	starpu_omp_proc_bind_spread = 4       /**< Assign team threads as a sparse distribution over the selected places.*/
};

/**
   Set of attributes used for creating a new parallel region.
   \sa starpu_omp_parallel_region()
*/
struct starpu_omp_parallel_region_attr
{
	/**
	   ::starpu_codelet (\ref API_Codelet_And_Tasks) to use for the
	   parallel region implicit tasks. The codelet must provide a
	   CPU implementation function.
	*/
	struct starpu_codelet  cl;
	/**
	   Array of zero or more ::starpu_data_handle_t data handle to
	   be passed to the parallel region implicit tasks.
	*/
	starpu_data_handle_t  *handles;
	/**
	   Optional pointer to an inline argument to be passed to the
	   region implicit tasks.
	*/
	void     *cl_arg;
	/**
	   Size of the optional inline argument to be passed to the
	   region implicit tasks, or 0 if unused.
	*/
	size_t    cl_arg_size;
	/**
	    Boolean indicating whether the optional inline argument
	    should be automatically freed (true), or not (false).
	*/
	unsigned  cl_arg_free;

	/**
	   Boolean indicating whether the \b if clause of the
	   corresponding <c>pragma omp parallel</c> is true or false.
	*/
	int if_clause;

	/**
	   Integer indicating the requested number of threads in the
	   team of the newly created parallel region, or 0 to let the
	   runtime choose the number of threads alone. This attribute
	   may be ignored by the runtime system if the requested
	   number of threads is higher than the number of threads that
	   the runtime can create.
	*/
	int num_threads;
};

/**
   Set of attributes used for creating a new task region.
   \sa starpu_omp_task_region()
*/
struct starpu_omp_task_region_attr
{
	/**
	   ::starpu_codelet (\ref API_Codelet_And_Tasks) to use for
	   the task region explicit task. The codelet must provide a
	   CPU implementation function or an accelerator
	   implementation for offloaded target regions.
	*/
	struct starpu_codelet  cl;
	/**
	   Array of zero or more ::starpu_data_handle_t data handle to
	   be passed to the task region explicit tasks.
	*/
	starpu_data_handle_t  *handles;
	/**
	   Optional pointer to an inline argument to be passed to the
	   region implicit tasks.
	*/
	void     *cl_arg;
	/**
	   Size of the optional inline argument to be passed to the
	   region implicit tasks, or 0 if unused.
	*/
	size_t    cl_arg_size;
	/**
	   Boolean indicating whether the optional inline argument
	   should be automatically freed (true), or not (false).
	*/
	unsigned  cl_arg_free;
	int       priority;

	/**
	   Boolean indicating whether the \b if clause of the
	   corresponding <c>pragma omp task</c> is true or false.
	*/
	int if_clause;
	/**
	   Boolean indicating whether the \b final clause of the
	   corresponding <c>pragma omp task</c> is true or false.
	*/
	int final_clause;

	/**
	    Boolean indicating whether the \b untied clause of the
	    corresponding <c>pragma omp task</c> is true or false.
	*/
	int untied_clause;
	/**
	   Boolean indicating whether the \b mergeable clause of the
	   corresponding <c>pragma omp task</c> is true or false.
	*/
	int mergeable_clause;

	/**
	   taskloop attribute
	*/
	int is_loop;
	int nogroup_clause;

	int collapse;
	int num_tasks;
	unsigned long long nb_iterations;
	unsigned long long grainsize;
	unsigned long long begin_i;
	unsigned long long end_i;
	unsigned long long chunk;
};

#ifdef __cplusplus
extern "C"
{
#define __STARPU_OMP_NOTHROW throw ()
#else
#define __STARPU_OMP_NOTHROW __attribute__((__nothrow__))
#endif

/**
   @name Initialisation
   @{
*/

/**
   Initialize StarPU and its OpenMP Runtime support.
*/
extern int starpu_omp_init(void) __STARPU_OMP_NOTHROW;
/**
   Shutdown StarPU and its OpenMP Runtime support.
*/
extern void starpu_omp_shutdown(void) __STARPU_OMP_NOTHROW;

/** @} */

/**
   @name Parallel
   \anchor ORS_Parallel
   @{
*/

/**
   Generate and launch an OpenMP parallel region and return after its
   completion. \p attr specifies the attributes for the generated parallel region.
   If this function is called from inside another, generating, parallel region, the
   generated parallel region is nested within the generating parallel region.

   This function can be used to implement <c>\#pragma omp parallel</c>.
*/
extern void starpu_omp_parallel_region(const struct starpu_omp_parallel_region_attr *attr) __STARPU_OMP_NOTHROW;

/**
   Execute a function only on the master thread of the OpenMP
   parallel region it is called from. When called from a thread that is not the
   master of the parallel region it is called from, this function does nothing. \p
   f is the function to be called. \p arg is an argument passed to function \p f.

   This function can be used to implement <c>\#pragma omp master</c>.
*/
extern void starpu_omp_master(void (*f)(void *arg), void *arg) __STARPU_OMP_NOTHROW;

/**
   Determine whether the calling thread is the master of the OpenMP parallel region
   it is called from or not.

   This function can be used to implement <c>\#pragma omp master</c> without code
   outlining.
   \return <c>!0</c> if called by the region's master thread.
   \return <c>0</c> if not called by the region's master thread.
*/
extern int starpu_omp_master_inline(void) __STARPU_OMP_NOTHROW;

/** @} */

/**
   @name Synchronization
   \anchor ORS_Synchronization
   @{
*/

/**
   Wait until each participating thread of the innermost OpenMP parallel region
   has reached the barrier and each explicit OpenMP task bound to this region has
   completed its execution.

   This function can be used to implement <c>\#pragma omp barrier</c>.
*/
extern void starpu_omp_barrier(void) __STARPU_OMP_NOTHROW;

/**
   Wait until no other thread is executing within the context of the selected
   critical section, then proceeds to the exclusive execution of a function within
   the critical section. \p f is the function to be executed in the critical
   section. \p arg is an argument passed to function \p f. \p name is the name of
   the selected critical section. If <c>name == NULL</c>, the selected critical
   section is the unique anonymous critical section.

   This function can be used to implement <c>\#pragma omp
   critical</c>.
*/
extern void starpu_omp_critical(void (*f)(void *arg), void *arg, const char *name) __STARPU_OMP_NOTHROW;

/**
   Wait until execution can proceed exclusively within the context of the
   selected critical section. \p name is the name of the selected critical
   section. If <c>name == NULL</c>, the selected critical section is the unique
   anonymous critical section.

   This function together with #starpu_omp_critical_inline_end can be used to
   implement <c>\#pragma omp critical</c> without code outlining.
*/
extern void starpu_omp_critical_inline_begin(const char *name) __STARPU_OMP_NOTHROW;

/**
   End the exclusive execution within the context of the selected critical
   section. \p name is the name of the selected critical section. If
   <c>name==NULL</c>, the selected critical section is the unique anonymous
   critical section.

   This function together with #starpu_omp_critical_inline_begin can be used to
   implement <c>\#pragma omp critical</c> without code outlining.
*/
extern void starpu_omp_critical_inline_end(const char *name) __STARPU_OMP_NOTHROW;

/** @} */

/**
   @name Worksharing
   \anchor ORS_Worksharing
   @{
*/

/**
   Ensure that a single participating thread of the innermost OpenMP parallel
   region executes a function. \p f is the function to be executed by a single
   thread. \p arg is an argument passed to function \p f. \p nowait is a flag
   indicating whether an implicit barrier is requested after the single section
   (<c>nowait==0</c>) or not (<c>nowait==!0</c>).

   This function can be used to implement <c>\#pragma omp single</c>.
*/
extern void starpu_omp_single(void (*f)(void *arg), void *arg, int nowait) __STARPU_OMP_NOTHROW;

/**
   Decide whether the current thread is elected to run the following single
   section among the participating threads of the innermost OpenMP parallel
   region.

   This function can be used to implement <c>\#pragma omp single</c> without code
   outlining.
   \return <c>!0</c> if the calling thread has won the election.
   \return <c>0</c> if the calling thread has lost the election.
*/
extern int starpu_omp_single_inline(void) __STARPU_OMP_NOTHROW;

/**
   Execute \p f on a single task of the current parallel region
   task, and then broadcast the contents of the memory block pointed by the
   copyprivate pointer \p data and of size \p data_size to the corresponding \p
   data pointed memory blocks of all the other participating region tasks. This
   function can be used to implement <c>\#pragma omp single</c> with a copyprivate
   clause.

   \sa starpu_omp_single_copyprivate_inline
   \sa starpu_omp_single_copyprivate_inline_begin
   \sa starpu_omp_single_copyprivate_inline_end
*/
extern void starpu_omp_single_copyprivate(void (*f)(void *arg, void *data, unsigned long long data_size), void *arg, void *data, unsigned long long data_size) __STARPU_OMP_NOTHROW;

/**
   Elect one task among the tasks of the current parallel region
   task to execute the following single section, and then broadcast the
   copyprivate pointer \p data to all the other participating region tasks. This
   function can be used to implement <c>\#pragma omp single</c> with a copyprivate
   clause without code outlining.

   \sa starpu_omp_single_copyprivate_inline
   \sa starpu_omp_single_copyprivate_inline_end
*/
extern void *starpu_omp_single_copyprivate_inline_begin(void *data) __STARPU_OMP_NOTHROW;

/**
   Complete the execution of a single section and return the
   broadcasted copyprivate pointer for tasks that lost the election and <c>NULL</c> for
   the task that won the election. This function can be used to implement
   <c>\#pragma omp single</c> with a copyprivate clause without code outlining.

   \return the copyprivate pointer for tasks that lost the election and therefore did not execute the code of the single section.
   \return <c>NULL</c> for the task that won the election and executed the code of the single section.

   \sa starpu_omp_single_copyprivate_inline
   \sa starpu_omp_single_copyprivate_inline_begin
*/
extern void starpu_omp_single_copyprivate_inline_end(void) __STARPU_OMP_NOTHROW;

/**
   Execute a parallel loop together with the other threads participating to the
   innermost parallel region. \p f is the function to be executed iteratively. \p
   arg is an argument passed to function \p f. \p nb_iterations is the number of
   iterations to be performed by the parallel loop. \p chunk is the number of
   consecutive iterations that should be affected to the same thread when
   scheduling the loop workshares, it follows the semantics of the \c modifier
   argument in OpenMP <c>\#pragma omp for</c> specification. \p schedule is the
   scheduling mode according to the OpenMP specification. \p ordered is a flag
   indicating whether the loop region may contain an ordered section
   (<c>ordered==!0</c>) or not (<c>ordered==0</c>). \p nowait is a flag
   indicating whether an implicit barrier is requested after the for section
   (<c>nowait==0</c>) or not (<c>nowait==!0</c>).

   The function \p f will be called with arguments \p _first_i, the first iteration
   to perform, \p _nb_i, the number of consecutive iterations to perform before
   returning, \p arg, the free \p arg argument.

   This function can be used to implement <c>\#pragma omp for</c>.
*/
extern void starpu_omp_for(void (*f)(unsigned long long _first_i, unsigned long long _nb_i, void *arg), void *arg, unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, int nowait) __STARPU_OMP_NOTHROW;

/**
   Decide whether the current thread should start to execute a parallel loop
   section. See #starpu_omp_for for the argument description.

   This function together with #starpu_omp_for_inline_next can be used to
   implement <c>\#pragma omp for</c> without code outlining.

   \return <c>!0</c> if the calling thread participates to the loop region and
   should execute a first chunk of iterations. In that case, \p *_first_i will be
   set to the first iteration of the chunk to perform and \p *_nb_i will be set to
   the number of iterations of the chunk to perform.

   \return <c>0</c> if the calling thread does not participate to the loop region
   because all the available iterations have been affected to the other threads of
   the parallel region.

   \sa starpu_omp_for
*/
extern int starpu_omp_for_inline_first(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_first_i, unsigned long long *_nb_i) __STARPU_OMP_NOTHROW;

/**
   Decide whether the current thread should continue to execute a parallel loop
   section. See #starpu_omp_for for the argument description.

   This function together with #starpu_omp_for_inline_first can be used to
   implement <c>\#pragma omp for</c> without code outlining.

   \return <c>!0</c> if the calling thread should execute a next chunk of
   iterations. In that case, \p *_first_i will be set to the first iteration of the
   chunk to perform and \p *_nb_i will be set to the number of iterations of the
   chunk to perform.

   \return <c>0</c> if the calling thread does not participate anymore to the loop
   region because all the available iterations have been affected to the other
   threads of the parallel region.

   \sa starpu_omp_for
*/
extern int starpu_omp_for_inline_next(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_first_i, unsigned long long *_nb_i) __STARPU_OMP_NOTHROW;

/**
   Alternative implementation of a parallel loop. Differ from
   #starpu_omp_for in the expected arguments of the loop function \c f.

   The function \p f will be called with arguments \p _begin_i, the first iteration
   to perform, \p _end_i, the first iteration not to perform before
   returning, \p arg, the free \p arg argument.

   This function can be used to implement <c>\#pragma omp for</c>.

   \sa starpu_omp_for
*/
extern void starpu_omp_for_alt(void (*f)(unsigned long long _begin_i, unsigned long long _end_i, void *arg), void *arg, unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, int nowait) __STARPU_OMP_NOTHROW;

/**
   Inline version of the alternative implementation of a parallel loop.

   This function together with #starpu_omp_for_inline_next_alt can be used to
   implement <c>\#pragma omp for</c> without code outlining.

   \sa starpu_omp_for
   \sa starpu_omp_for_alt
   \sa starpu_omp_for_inline_first
*/
extern int starpu_omp_for_inline_first_alt(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_begin_i, unsigned long long *_end_i) __STARPU_OMP_NOTHROW;

/**
   Inline version of the alternative implementation of a parallel loop.

   This function together with #starpu_omp_for_inline_first_alt can be used to
   implement <c>\#pragma omp for</c> without code outlining.

   \sa starpu_omp_for
   \sa starpu_omp_for_alt
   \sa starpu_omp_for_inline_next
*/
extern int starpu_omp_for_inline_next_alt(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_begin_i, unsigned long long *_end_i) __STARPU_OMP_NOTHROW;

/**
   Ensure that a function is sequentially executed once for each iteration in
   order within a parallel loop, by the thread that own the iteration. \p f is the
   function to be executed by the thread that own the current iteration. \p arg is
   an argument passed to function \p f.

   This function can be used to implement <c>\#pragma omp ordered</c>.
*/
extern void starpu_omp_ordered(void (*f)(void *arg), void *arg) __STARPU_OMP_NOTHROW;

/**
   Wait until all the iterations of a parallel loop below the iteration owned by
   the current thread have been executed.

   This function together with #starpu_omp_ordered_inline_end can be used to
   implement <c>\#pragma omp ordered</c> without code code outlining.
*/
extern void starpu_omp_ordered_inline_begin(void) __STARPU_OMP_NOTHROW;

/**
   Notify that the ordered section for the current iteration has been completed.

   This function together with #starpu_omp_ordered_inline_begin can be used to
   implement <c>\#pragma omp ordered</c> without code code outlining.
*/
extern void starpu_omp_ordered_inline_end(void) __STARPU_OMP_NOTHROW;

/**
   Ensure that each function of a given array of functions is executed by one and
   only one thread. \p nb_sections is the number of functions in the array \p
   section_f. \p section_f is the array of functions to be executed as sections. \p
   section_arg is an array of arguments to be passed to the corresponding function.
   \p nowait is a flag indicating whether an implicit barrier is requested after
   the execution of all the sections (<c>nowait==0</c>) or not (<c>nowait==!0</c>).

   This function can be used to implement <c>\#pragma omp sections</c> and <c>\#pragma omp section</c>.
 */
extern void starpu_omp_sections(unsigned long long nb_sections, void (**section_f)(void *arg), void **section_arg, int nowait) __STARPU_OMP_NOTHROW;

/**
   Alternative implementation of sections. Differ from
   #starpu_omp_sections in that all the sections are combined within a single
   function in this version. \p section_f is the function implementing the combined
   sections.

   The function \p section_f will be called with arguments \p section_num, the
   section number to be executed, \p arg, the entry of \p section_arg corresponding
   to this section.

   This function can be used to implement <c>\#pragma omp sections</c> and <c>\#pragma omp section</c>.

   \sa starpu_omp_sections
 */
extern void starpu_omp_sections_combined(unsigned long long nb_sections, void (*section_f)(unsigned long long section_num, void *arg), void *section_arg, int nowait) __STARPU_OMP_NOTHROW;

/** @} */

/**
   @name Task
   \anchor ORS_Task
   @{
*/

/**
   Generate an explicit child task. The execution of the generated task is
   asynchronous with respect to the calling code unless specified otherwise.
   \p attr specifies the attributes for the generated task region.

   This function can be used to implement <c>\#pragma omp task</c>.
 */
extern void starpu_omp_task_region(const struct starpu_omp_task_region_attr *attr) __STARPU_OMP_NOTHROW;

/**
   Wait for the completion of the tasks generated by the current task. This
   function does not wait for the descendants of the tasks generated by the current
   task.

   This function can be used to implement <c>\#pragma omp taskwait</c>.
 */
extern void starpu_omp_taskwait(void) __STARPU_OMP_NOTHROW;

/**
   Launch a function and wait for the completion of every descendant task
   generated during the execution of the function.

   This function can be used to implement <c>\#pragma omp taskgroup</c>.

   \sa starpu_omp_taskgroup_inline_begin
   \sa starpu_omp_taskgroup_inline_end
 */
extern void starpu_omp_taskgroup(void (*f)(void *arg), void *arg) __STARPU_OMP_NOTHROW;

/**
   Launch a function and gets ready to wait for the completion of every descendant task
   generated during the dynamic scope of the taskgroup.

   This function can be used to implement <c>\#pragma omp taskgroup</c> without code outlining.

   \sa starpu_omp_taskgroup
   \sa starpu_omp_taskgroup_inline_end
 */
extern void starpu_omp_taskgroup_inline_begin(void) __STARPU_OMP_NOTHROW;

/**
   Wait for the completion of every descendant task
   generated during the dynamic scope of the taskgroup.

   This function can be used to implement <c>\#pragma omp taskgroup</c> without code outlining.

   \sa starpu_omp_taskgroup
   \sa starpu_omp_taskgroup_inline_begin
 */
extern void starpu_omp_taskgroup_inline_end(void) __STARPU_OMP_NOTHROW;

extern void starpu_omp_taskloop_inline_begin(struct starpu_omp_task_region_attr *attr) __STARPU_OMP_NOTHROW;

extern void starpu_omp_taskloop_inline_end(const struct starpu_omp_task_region_attr *attr) __STARPU_OMP_NOTHROW;

/** @} */

/**
   @name API
   \anchor ORS_API
   @{
*/

/**
   Set ICVS nthreads_var for the parallel regions to be created
   with the current region.

   Note: The StarPU OpenMP runtime support currently ignores
   this setting for nested parallel regions.

   \sa starpu_omp_get_num_threads
   \sa starpu_omp_get_thread_num
   \sa starpu_omp_get_max_threads
   \sa starpu_omp_get_num_procs
*/
extern void starpu_omp_set_num_threads(int threads) __STARPU_OMP_NOTHROW;

/**
   Return the number of threads of the current region.

   \return the number of threads of the current region.

   \sa starpu_omp_set_num_threads
   \sa starpu_omp_get_thread_num
   \sa starpu_omp_get_max_threads
   \sa starpu_omp_get_num_procs
 */
extern int starpu_omp_get_num_threads() __STARPU_OMP_NOTHROW;

/**
   Return the rank of the current thread among the threads
   of the current region.

   \return the rank of the current thread in the current region.

   \sa starpu_omp_set_num_threads
   \sa starpu_omp_get_num_threads
   \sa starpu_omp_get_max_threads
   \sa starpu_omp_get_num_procs
 */
extern int starpu_omp_get_thread_num() __STARPU_OMP_NOTHROW;

/**
   Return the maximum number of threads that can be used to
   create a region from the current region.

   \return the maximum number of threads that can be used to create a region from the current region.

   \sa starpu_omp_set_num_threads
   \sa starpu_omp_get_num_threads
   \sa starpu_omp_get_thread_num
   \sa starpu_omp_get_num_procs
 */
extern int starpu_omp_get_max_threads() __STARPU_OMP_NOTHROW;

/**
   Return the number of StarPU CPU workers.

   \return the number of StarPU CPU workers.

   \sa starpu_omp_set_num_threads
   \sa starpu_omp_get_num_threads
   \sa starpu_omp_get_thread_num
   \sa starpu_omp_get_max_threads
*/
extern int starpu_omp_get_num_procs(void) __STARPU_OMP_NOTHROW;

/**
   Return whether it is called from the scope of a parallel region or not.

   \return <c>!0</c> if called from a parallel region scope.
   \return <c>0</c> otherwise.
*/
extern int starpu_omp_in_parallel(void) __STARPU_OMP_NOTHROW;

/**
   Enable (1) or disable (0) dynamically adjusting the number of parallel threads.

   Note: The StarPU OpenMP runtime support currently ignores the argument of this function.

   \sa starpu_omp_get_dynamic
*/
extern void starpu_omp_set_dynamic(int dynamic_threads) __STARPU_OMP_NOTHROW;

/**
   Return the state of dynamic thread number adjustment.

   \return <c>!0</c> if dynamic thread number adjustment is enabled.
   \return <c>0</c> otherwise.

   \sa starpu_omp_set_dynamic
*/
extern int starpu_omp_get_dynamic(void) __STARPU_OMP_NOTHROW;

/**
   Enable (1) or disable (0) nested parallel regions.

   Note: The StarPU OpenMP runtime support currently ignores the argument of this function.

   \sa starpu_omp_get_nested
   \sa starpu_omp_get_max_active_levels
   \sa starpu_omp_set_max_active_levels
   \sa starpu_omp_get_level
   \sa starpu_omp_get_active_level
*/
extern void starpu_omp_set_nested(int nested) __STARPU_OMP_NOTHROW;

/**
   Return whether nested parallel sections are enabled or not.

   \return <c>!0</c> if nested parallel sections are enabled.
   \return <c>0</c> otherwise.

   \sa starpu_omp_set_nested
   \sa starpu_omp_get_max_active_levels
   \sa starpu_omp_set_max_active_levels
   \sa starpu_omp_get_level
   \sa starpu_omp_get_active_level
*/
extern int starpu_omp_get_nested(void) __STARPU_OMP_NOTHROW;

/**
   Return the state of the cancel ICVS var.
 */
extern int starpu_omp_get_cancellation(void) __STARPU_OMP_NOTHROW;

/**
   Set the default scheduling kind for upcoming loops within the
   current parallel section. \p kind is the scheduler kind, \p modifier
   complements the scheduler kind with informations such as the chunk size,
   in accordance with the OpenMP specification.

   \sa starpu_omp_get_schedule
 */
extern void starpu_omp_set_schedule(enum starpu_omp_sched_value kind, int modifier) __STARPU_OMP_NOTHROW;

/**
   Return the current selected default loop scheduler.

   \return the kind and the modifier of the current default loop scheduler.

   \sa starpu_omp_set_schedule
*/
extern void starpu_omp_get_schedule(enum starpu_omp_sched_value *kind, int *modifier) __STARPU_OMP_NOTHROW;

/**
   Return the number of StarPU CPU workers.

   \return the number of StarPU CPU workers.
*/
extern int starpu_omp_get_thread_limit(void) __STARPU_OMP_NOTHROW;

/**
   Set the maximum number of allowed active parallel section levels.

   Note: The StarPU OpenMP runtime support currently ignores the argument of this function and assume \p max_levels equals <c>1</c> instead.

   \sa starpu_omp_set_nested
   \sa starpu_omp_get_nested
   \sa starpu_omp_get_max_active_levels
   \sa starpu_omp_get_level
   \sa starpu_omp_get_active_level
*/
extern void starpu_omp_set_max_active_levels(int max_levels) __STARPU_OMP_NOTHROW;

/**
   Return the current maximum number of allowed active parallel section levels

   \return the current maximum number of allowed active parallel section levels.

   \sa starpu_omp_set_nested
   \sa starpu_omp_get_nested
   \sa starpu_omp_set_max_active_levels
   \sa starpu_omp_get_level
   \sa starpu_omp_get_active_level
*/
extern int starpu_omp_get_max_active_levels(void) __STARPU_OMP_NOTHROW;

/**
   Return the nesting level of the current parallel section.

   \return the nesting level of the current parallel section.

   \sa starpu_omp_set_nested
   \sa starpu_omp_get_nested
   \sa starpu_omp_get_max_active_levels
   \sa starpu_omp_set_max_active_levels
   \sa starpu_omp_get_active_level
*/
extern int starpu_omp_get_level(void) __STARPU_OMP_NOTHROW;

/**
   Return the number of the ancestor of the current parallel section.

   \return the number of the ancestor of the current parallel section.
*/
extern int starpu_omp_get_ancestor_thread_num(int level) __STARPU_OMP_NOTHROW;

/**
   Return the size of the team of the current parallel section.

   \return the size of the team of the current parallel section.
*/
extern int starpu_omp_get_team_size(int level) __STARPU_OMP_NOTHROW;

/**
   Return the nestinglevel of the current innermost active parallel section.

   \return the nestinglevel of the current innermost active parallel section.

   \sa starpu_omp_set_nested
   \sa starpu_omp_get_nested
   \sa starpu_omp_get_max_active_levels
   \sa starpu_omp_set_max_active_levels
   \sa starpu_omp_get_level
*/
extern int starpu_omp_get_active_level(void) __STARPU_OMP_NOTHROW;

/**
   Check whether the current task is final or not.

   \return <c>!0</c> if called from a final task.
   \return <c>0</c> otherwise.
*/
extern int starpu_omp_in_final(void) __STARPU_OMP_NOTHROW;

/**
   Return the proc_bind setting of the current parallel region.

   \return the proc_bind setting of the current parallel region.
*/
extern enum starpu_omp_proc_bind_value starpu_omp_get_proc_bind(void) __STARPU_OMP_NOTHROW;

extern int starpu_omp_get_num_places(void) __STARPU_OMP_NOTHROW;

extern int starpu_omp_get_place_num_procs(int place_num) __STARPU_OMP_NOTHROW;

extern void starpu_omp_get_place_proc_ids(int place_num, int *ids) __STARPU_OMP_NOTHROW;

extern int starpu_omp_get_place_num(void) __STARPU_OMP_NOTHROW;

extern int starpu_omp_get_partition_num_places(void) __STARPU_OMP_NOTHROW;

extern void starpu_omp_get_partition_place_nums(int *place_nums) __STARPU_OMP_NOTHROW;

/**
   Set the number of the device to use as default.

   Note: The StarPU OpenMP runtime support currently ignores the argument of this function.

   \sa starpu_omp_get_default_device
   \sa starpu_omp_is_initial_device
*/
extern void starpu_omp_set_default_device(int device_num) __STARPU_OMP_NOTHROW;

/**
   Return the number of the device used as default.

   \return the number of the device used as default.

   \sa starpu_omp_set_default_device
   \sa starpu_omp_is_initial_device
 */
extern int starpu_omp_get_default_device(void) __STARPU_OMP_NOTHROW;

/**
   Return the number of the devices.

   \return the number of the devices.
*/
extern int starpu_omp_get_num_devices(void) __STARPU_OMP_NOTHROW;

/**
   Return the number of teams in the current teams region.

   \return the number of teams in the current teams region.

   \sa starpu_omp_get_num_teams
*/
extern int starpu_omp_get_num_teams(void) __STARPU_OMP_NOTHROW;

/**
   Return the team number of the calling thread.

   \return the team number of the calling thread.

   \sa starpu_omp_get_num_teams
*/
extern int starpu_omp_get_team_num(void) __STARPU_OMP_NOTHROW;

/**
   Check whether the current device is the initial device or not.
*/
extern int starpu_omp_is_initial_device(void) __STARPU_OMP_NOTHROW;

/**
 */
extern int starpu_omp_get_initial_device(void) __STARPU_OMP_NOTHROW;

/**
   Return the maximum value that can be specified in the priority
   clause.

   \return <c>!0</c> if called from the host device.
   \return <c>0</c> otherwise.

   \sa starpu_omp_set_default_device
   \sa starpu_omp_get_default_device
*/
extern int starpu_omp_get_max_task_priority(void) __STARPU_OMP_NOTHROW;

/**
   Initialize an opaque lock object.

   \sa starpu_omp_destroy_lock
   \sa starpu_omp_set_lock
   \sa starpu_omp_unset_lock
   \sa starpu_omp_test_lock
*/
extern void starpu_omp_init_lock(starpu_omp_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Destroy an opaque lock object.

   \sa starpu_omp_init_lock
   \sa starpu_omp_set_lock
   \sa starpu_omp_unset_lock
   \sa starpu_omp_test_lock
*/
extern void starpu_omp_destroy_lock(starpu_omp_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Lock an opaque lock object. If the lock is already locked, the
   function will block until it succeeds in exclusively acquiring the lock.

   \sa starpu_omp_init_lock
   \sa starpu_omp_destroy_lock
   \sa starpu_omp_unset_lock
   \sa starpu_omp_test_lock
*/
extern void starpu_omp_set_lock(starpu_omp_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Unlock a previously locked lock object. The behaviour of this
   function is unspecified if it is called on an unlocked lock object.

   \sa starpu_omp_init_lock
   \sa starpu_omp_destroy_lock
   \sa starpu_omp_set_lock
   \sa starpu_omp_test_lock
*/
extern void starpu_omp_unset_lock(starpu_omp_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Unblockingly attempt to lock a lock object and return whether
   it succeeded or not.

   \return <c>!0</c> if the function succeeded in acquiring the lock.
   \return <c>0</c> if the lock was already locked.

   \sa starpu_omp_init_lock
   \sa starpu_omp_destroy_lock
   \sa starpu_omp_set_lock
   \sa starpu_omp_unset_lock
*/
extern int starpu_omp_test_lock(starpu_omp_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Initialize an opaque lock object supporting nested locking operations.

   \sa starpu_omp_destroy_nest_lock
   \sa starpu_omp_set_nest_lock
   \sa starpu_omp_unset_nest_lock
   \sa starpu_omp_test_nest_lock
*/
extern void starpu_omp_init_nest_lock(starpu_omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Destroy an opaque lock object supporting nested locking operations.

   \sa starpu_omp_init_nest_lock
   \sa starpu_omp_set_nest_lock
   \sa starpu_omp_unset_nest_lock
   \sa starpu_omp_test_nest_lock
*/
extern void starpu_omp_destroy_nest_lock(starpu_omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Lock an opaque lock object supporting nested locking operations.
   If the lock is already locked by another task, the function will block until
   it succeeds in exclusively acquiring the lock. If the lock is already taken by
   the current task, the function will increase the nested locking level of the
   lock object.

   \sa starpu_omp_init_nest_lock
   \sa starpu_omp_destroy_nest_lock
   \sa starpu_omp_unset_nest_lock
   \sa starpu_omp_test_nest_lock
*/
extern void starpu_omp_set_nest_lock(starpu_omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Unlock a previously locked lock object supporting nested locking
   operations. If the lock has been locked multiple times in nested fashion, the
   nested locking level is decreased and the lock remains locked. Otherwise, if
   the lock has only been locked once, it becomes unlocked. The behaviour of this
   function is unspecified if it is called on an unlocked lock object. The
   behaviour of this function is unspecified if it is called from a different task
   than the one that locked the lock object.

   \sa starpu_omp_init_nest_lock
   \sa starpu_omp_destroy_nest_lock
   \sa starpu_omp_set_nest_lock
   \sa starpu_omp_test_nest_lock
*/
extern void starpu_omp_unset_nest_lock(starpu_omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Unblocking attempt to lock an opaque lock object supporting
   nested locking operations and returns whether it succeeded or not. If the lock
   is already locked by another task, the function will return without having
   acquired the lock. If the lock is already taken by the current task, the
   function will increase the nested locking level of the lock object.

   \return <c>!0</c> if the function succeeded in acquiring the lock.
   \return <c>0</c> if the lock was already locked.

   \sa starpu_omp_init_nest_lock
   \sa starpu_omp_destroy_nest_lock
   \sa starpu_omp_set_nest_lock
   \sa starpu_omp_unset_nest_lock
*/
extern int starpu_omp_test_nest_lock(starpu_omp_nest_lock_t *lock) __STARPU_OMP_NOTHROW;

/**
   Implement the entry point of a fallback global atomic region.
   Block until it succeeds in acquiring exclusive access to the global atomic
   region.

   \sa starpu_omp_atomic_fallback_inline_end
 */
extern void starpu_omp_atomic_fallback_inline_begin(void) __STARPU_OMP_NOTHROW;

/**
   Implement the exit point of a fallback global atomic region.
   Release the exclusive access to the global atomic region.

   \sa starpu_omp_atomic_fallback_inline_begin
 */
extern void starpu_omp_atomic_fallback_inline_end(void) __STARPU_OMP_NOTHROW;

/**
   Return the elapsed wallclock time in seconds.

   \return the elapsed wallclock time in seconds.

   \sa starpu_omp_get_wtick
*/
extern double starpu_omp_get_wtime(void) __STARPU_OMP_NOTHROW;

/**
   Return the precision of the time used by \p starpu_omp_get_wtime().

   \return the precision of the time used by \p starpu_omp_get_wtime().

   \sa starpu_omp_get_wtime
*/
extern double starpu_omp_get_wtick(void) __STARPU_OMP_NOTHROW;

/**
   Enable setting additional vector metadata needed by the OpenMP Runtime Support.

   \p handle is vector data handle.
   \p slice_base is the base of an array slice, expressed in number of vector elements from the array base.

   \sa STARPU_VECTOR_GET_SLICE_BASE
 */
extern void starpu_omp_vector_annotate(starpu_data_handle_t handle, uint32_t slice_base) __STARPU_OMP_NOTHROW;

/**
 */
extern struct starpu_arbiter *starpu_omp_get_default_arbiter(void) __STARPU_OMP_NOTHROW;

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_OPENMP && !STARPU_DONT_INCLUDE_OPENMP_HEADERS */

/** @} */

#endif /* __STARPU_OPENMP_H__ */
