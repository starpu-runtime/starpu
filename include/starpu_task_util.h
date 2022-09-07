/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_TASK_UTIL_H__
#define __STARPU_TASK_UTIL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Insert_Task Task Insert Utility
   @{
*/

/* NOTE: when adding a value here, please make sure to update both
 * src/util/starpu_task_insert_utils.c (in two places) and
 * mpi/src/starpu_mpi_task_insert.c and mpi/src/starpu_mpi_task_insert_fortran.c */

#define STARPU_MODE_SHIFT 17

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a constant value and the size of the constant
 */
#define STARPU_VALUE (1 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a callback function
*/
#define STARPU_CALLBACK (2 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by two
   pointers: one to a callback function, and the other to be given as
   an argument to the callback function; this is equivalent to using
   both ::STARPU_CALLBACK and ::STARPU_CALLBACK_ARG.
*/
#define STARPU_CALLBACK_WITH_ARG (3 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to be given as an argument to the callback function
*/
#define STARPU_CALLBACK_ARG (4 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must
   be followed by a integer defining a priority level
*/
#define STARPU_PRIORITY (5 << STARPU_MODE_SHIFT)

/**
   \ingroup API_MPI_Support
   Used when calling starpu_mpi_task_insert(), must be followed by a
   integer value which specified the node on which to execute the
   codelet.
 */
#define STARPU_EXECUTE_ON_NODE (6 << STARPU_MODE_SHIFT)

/**
   \ingroup API_MPI_Support
   Used when calling starpu_mpi_task_insert(), must be followed by a
   data handle to specify that the node owning the given data will
   execute the codelet.
*/
#define STARPU_EXECUTE_ON_DATA (7 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an array of
   handles and the number of elements in the array (as int). This is equivalent
   to passing the handles as separate parameters with ::STARPU_R,
   ::STARPU_W or ::STARPU_RW.
*/
#define STARPU_DATA_ARRAY (8 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an array of
   struct starpu_data_descr and the number of elements in the array (as int).
   This is equivalent to passing the handles with the corresponding modes.
*/
#define STARPU_DATA_MODE_ARRAY (9 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a tag.
*/
#define STARPU_TAG (10 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a tag.
*/
#define STARPU_HYPERVISOR_TAG (11 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   amount of floating point operations, as a double. Users <b>MUST</b>
   explicitly cast into double, otherwise parameter passing will not
   work.
*/
#define STARPU_FLOPS (12 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by the id
   of the scheduling context to which to submit the task to.
*/
#define STARPU_SCHED_CTX (13 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a prologue callback function
*/
#define STARPU_PROLOGUE_CALLBACK (14 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to be given as an argument to the prologue callback
   function
*/
#define STARPU_PROLOGUE_CALLBACK_ARG (15 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a prologue callback pop function
*/
#define STARPU_PROLOGUE_CALLBACK_POP (16 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to be given as an argument to the prologue callback pop
   function
*/
#define STARPU_PROLOGUE_CALLBACK_POP_ARG (17 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   integer value specifying the worker on which to execute the task
   (as specified by starpu_task::execute_on_a_specific_worker)
*/
#define STARPU_EXECUTE_ON_WORKER (18 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   unsigned long long value specifying the mask of worker on which to execute
   the task (as specified by starpu_task::where)
*/
#define STARPU_EXECUTE_WHERE (19 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a tag
   stored in starpu_task::tag_id. Leave starpu_task::use_tag as 0.
*/
#define STARPU_TAG_ONLY (20 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an unsigned
   stored in starpu_task::possibly_parallel.
*/
#define STARPU_POSSIBLY_PARALLEL (21 << STARPU_MODE_SHIFT)

/**
   used when calling starpu_task_insert(), must be
   followed by an integer value specifying the worker order in which
   to execute the tasks (as specified by starpu_task::workerorder)
*/
#define STARPU_WORKER_ORDER (22 << STARPU_MODE_SHIFT)

/**
   \ingroup API_MPI_Support
   Used when calling starpu_mpi_task_insert(), must be followed by a
   identifier to a node selection policy. This is needed when several
   nodes own data in ::STARPU_W mode.
*/
#define STARPU_NODE_SELECTION_POLICY (23 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   char * stored in starpu_task::name.
*/
#define STARPU_NAME (24 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   memory buffer containing the arguments to be given to the task, and
   by the size of the arguments. The memory buffer should be the
   result of a previous call to starpu_codelet_pack_args(), and will
   be freed (i.e. starpu_task::cl_arg_free will be set to 1)
*/
#define STARPU_CL_ARGS (25 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), similarly to
   ::STARPU_CL_ARGS, must be followed by a memory buffer containing
   the arguments to be given to the task, and by the size of the
   arguments. The memory buffer should be the result of a previous
   call to starpu_codelet_pack_args(), and will NOT be freed (i.e.
   starpu_task::cl_arg_free will be set to 0)
*/
#define STARPU_CL_ARGS_NFREE (26 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   number of tasks as int, and an array containing these tasks. The
   function starpu_task_declare_deps_array() will be called with the
   given values.
*/
#define STARPU_TASK_DEPS_ARRAY (27 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   integer representing a color
*/
#define STARPU_TASK_COLOR (28 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   array of characters representing the sequential consistency for
   each buffer of the task.
*/
#define STARPU_HANDLES_SEQUENTIAL_CONSISTENCY (29 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   integer stating if the task is synchronous or not
*/
#define STARPU_TASK_SYNCHRONOUS (30 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   number of tasks as int, and an array containing these tasks. The
   function starpu_task_declare_end_deps_array() will be called with
   the given values.
*/
#define STARPU_TASK_END_DEPS_ARRAY (31 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   integer which will be given to starpu_task_end_dep_add()
*/
#define STARPU_TASK_END_DEP (32 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   unsigned being a number of workers, and an array of bits which size
   is the number of workers, the array indicates the set of workers
   which are allowed to execute the task.
*/
#define STARPU_TASK_WORKERIDS (33 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   unsigned which sets the sequential consistency for the data
   parameters of the task.
*/
#define STARPU_SEQUENTIAL_CONSISTENCY (34 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert() and alike, must be followed
   by a pointer to a struct starpu_profiling_task_info
 */
#define STARPU_TASK_PROFILING_INFO (35 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert() and alike, must be followed
   by an unsigned specifying not to allocate a submitorder id for the task
 */
#define STARPU_TASK_NO_SUBMITORDER (36 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), similarly to
   ::STARPU_CALLBACK_ARG, must be followed by a pointer to be given as
   an argument to the callback function, the argument will not be
   freed, i.e starpu_task::callback_arg_free will be set to 0
*/
#define STARPU_CALLBACK_ARG_NFREE (37 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), similarly to
   ::STARPU_CALLBACK_WITH_ARG, must be followed by two pointers: one
   to a callback function, and the other to be given as an argument to
   the callback function; this is equivalent to using both
   ::STARPU_CALLBACK and ::STARPU_CALLBACK_ARG_NFREE.
*/
#define STARPU_CALLBACK_WITH_ARG_NFREE (38 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), similarly to
   ::STARPU_PROLOGUE_CALLBACK_ARG, must be followed by a
   pointer to be given as an argument to the prologue callback
   function, the argument will not be
   freed, i.e starpu_task::prologue_callback_arg_free will be set to 0
*/
#define STARPU_PROLOGUE_CALLBACK_ARG_NFREE (39 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), similarly to
   ::STARPU_PROLOGUE_CALLBACK_POP_ARG, must be followed by a pointer
   to be given as an argument to the prologue callback pop function,
   the argument will not be freed, i.e
   starpu_task::prologue_callback_pop_arg_free will be set to 0
*/
#define STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE (40 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert() and alike, must be followed
   by a void* specifying the value to be set in starpu_task::sched_data
 */
#define STARPU_TASK_SCHED_DATA (41 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert() and alike, must be followed
   by a struct starpu_transaction * specifying the value to be set in
   the transaction field of the task.
 */
#define STARPU_TRANSACTION (42 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   char * stored in starpu_task::file.

   This is automatically set when FXT is enabled.
*/
#define STARPU_TASK_FILE (43 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by an
   int stored in starpu_task::line.

   This is automatically set when FXT is enabled.
*/
#define STARPU_TASK_LINE (44 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a epilogue callback function
*/
#define STARPU_EPILOGUE_CALLBACK (45 << STARPU_MODE_SHIFT)

/**
   Used when calling starpu_task_insert(), must be followed by a
   pointer to be given as an argument to the epilogue callback
   function
*/
#define STARPU_EPILOGUE_CALLBACK_ARG (46 << STARPU_MODE_SHIFT)

/**
   \ingroup API_Bubble
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a bubble decision function ::starpu_bubble_func_t
*/
#define STARPU_BUBBLE_FUNC (47 << STARPU_MODE_SHIFT)

/**
   \ingroup API_Bubble
   Used when calling starpu_task_insert(), must be followed by a
   pointer which will be passed to the function defined in
   starpu_codelet::bubble_func
*/
#define STARPU_BUBBLE_FUNC_ARG (48 << STARPU_MODE_SHIFT)

/**
   \ingroup API_Bubble
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a bubble DAG generation function
   ::starpu_bubble_gen_dag_func_t
*/
#define STARPU_BUBBLE_GEN_DAG_FUNC (49 << STARPU_MODE_SHIFT)

/**
   \ingroup API_Bubble
   Used when calling starpu_task_insert(), must be followed by a
   pointer which will be passed to the function defined in
   starpu_codelet::bubble_gen_dag_func
*/
#define STARPU_BUBBLE_GEN_DAG_FUNC_ARG (50 << STARPU_MODE_SHIFT)

/**
   \ingroup API_Bubble
   Used when calling starpu_task_insert(), must be followed by a
   pointer to a task. The task will be set as the bubble parent task
   when using the offline tracing tool.
*/
#define STARPU_BUBBLE_PARENT (51 << STARPU_MODE_SHIFT)

/**
   This has to be the last mode value plus 1
*/
#define STARPU_SHIFTED_MODE_MAX (52 << STARPU_MODE_SHIFT)

/**
   Set the given \p task corresponding to \p cl with the following arguments.
   The argument list must be zero-terminated. The arguments
   following the codelet are the same as the ones for the function
   starpu_task_insert().
   If some arguments of type ::STARPU_VALUE are given, the parameter
   starpu_task::cl_arg_free will be set to 1.
*/
int starpu_task_set(struct starpu_task *task, struct starpu_codelet *cl, ...);
#ifdef STARPU_USE_FXT
#define starpu_task_set(task, cl, ...) starpu_task_set((task), (cl), STARPU_TASK_FILE, __FILE__, STARPU_TASK_LINE, __LINE__, ##__VA_ARGS__)
#endif

/**
   Create a task corresponding to \p cl with the following arguments.
   The argument list must be zero-terminated. The arguments
   following the codelet are the same as the ones for the function
   starpu_task_insert().
   If some arguments of type ::STARPU_VALUE are given, the parameter
   starpu_task::cl_arg_free will be set to 1.
*/
struct starpu_task *starpu_task_build(struct starpu_codelet *cl, ...);
#ifdef STARPU_USE_FXT
#define starpu_task_build(cl, ...) starpu_task_build((cl), STARPU_TASK_FILE, __FILE__, STARPU_TASK_LINE, __LINE__, ##__VA_ARGS__)
#endif

/**
   Create and submit a task corresponding to \p cl with the following
   given arguments. The argument list must be zero-terminated.

   The arguments following the codelet can be of the following types:
   <ul>
   <li> ::STARPU_R, ::STARPU_W, ::STARPU_RW, ::STARPU_SCRATCH,
   ::STARPU_REDUX an access mode followed by a data handle;
   <li> ::STARPU_DATA_ARRAY followed by an array of data handles and
   its number of elements;
   <li> ::STARPU_DATA_MODE_ARRAY followed by an array of struct
   starpu_data_descr, i.e data handles with their associated access
   modes, and its number of elements;
   <li> ::STARPU_EXECUTE_ON_WORKER, ::STARPU_WORKER_ORDER followed by
   an integer value specifying the worker on which to execute the task
   (as specified by starpu_task::execute_on_a_specific_worker)
   <li> the specific values ::STARPU_VALUE, ::STARPU_CALLBACK,
   ::STARPU_CALLBACK_ARG, ::STARPU_CALLBACK_WITH_ARG,
   ::STARPU_PRIORITY, ::STARPU_TAG, ::STARPU_TAG_ONLY, ::STARPU_FLOPS,
   ::STARPU_SCHED_CTX, ::STARPU_CL_ARGS, ::STARPU_CL_ARGS_NFREE,
   ::STARPU_TASK_DEPS_ARRAY, ::STARPU_TASK_COLOR,
   ::STARPU_HANDLES_SEQUENTIAL_CONSISTENCY, ::STARPU_TASK_SYNCHRONOUS,
   ::STARPU_TASK_END_DEP followed by the appropriated objects as
   defined elsewhere.
   </ul>

   When using ::STARPU_DATA_ARRAY, the access mode of the data handles
   is not defined, it will be taken from the codelet
   starpu_codelet::modes or starpu_codelet::dyn_modes field. One
   should use ::STARPU_DATA_MODE_ARRAY to define the data handles
   along with the access modes.

   Parameters to be passed to the codelet implementation are defined
   through the type ::STARPU_VALUE. The function
   starpu_codelet_unpack_args() must be called within the codelet implementation to retrieve them.
*/
int starpu_task_insert(struct starpu_codelet *cl, ...);
#ifdef STARPU_USE_FXT
#define starpu_task_insert(cl, ...) starpu_task_insert((cl), STARPU_TASK_FILE, __FILE__, STARPU_TASK_LINE, __LINE__, ##__VA_ARGS__)
#endif

/**
   Similar to starpu_task_insert(). Kept to avoid breaking old codes.
*/
int starpu_insert_task(struct starpu_codelet *cl, ...);
#ifdef STARPU_USE_FXT
#define starpu_insert_task(cl, ...) starpu_insert_task((cl), STARPU_TASK_FILE, __FILE__, STARPU_TASK_LINE, __LINE__, ##__VA_ARGS__)
#endif

/**
   Assuming that there are already \p current_buffer data handles
   passed to the task, and if *allocated_buffers is not 0, the
   <c>task->dyn_handles</c> array has size \p *allocated_buffers, this
   function makes room for \p room other data handles, allocating or
   reallocating <c>task->dyn_handles</c> as necessary and updating \p
   *allocated_buffers accordingly. One can thus start with
   *allocated_buffers equal to 0 and current_buffer equal to 0, then
   make room by calling this function, then store handles with
   STARPU_TASK_SET_HANDLE(), make room again with this function, store
   yet more handles, etc.
*/
void starpu_task_insert_data_make_room(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int current_buffer, int room);

/**
   Store data handle \p handle into task \p task with mode \p
   arg_type, updating \p *allocated_buffers and \p *current_buffer
   accordingly.
*/
void starpu_task_insert_data_process_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int arg_type, starpu_data_handle_t handle);

/**
   Store \p nb_handles data handles \p handles into task \p task,
   updating \p *allocated_buffers and \p *current_buffer accordingly.
*/
void starpu_task_insert_data_process_array_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int nb_handles, starpu_data_handle_t *handles);

/**
   Store \p nb_descrs data handles described by \p descrs into task \p
   task, updating \p *allocated_buffers and \p *current_buffer
   accordingly.
*/
void starpu_task_insert_data_process_mode_array_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int nb_descrs, struct starpu_data_descr *descrs);

/**
   Pack arguments of type ::STARPU_VALUE into a buffer which can be
   given to a codelet and later unpacked with the function
   starpu_codelet_unpack_args().

   Instead of calling starpu_codelet_pack_args(), one can also call
   starpu_codelet_pack_arg_init(), then starpu_codelet_pack_arg() for
   each data, then starpu_codelet_pack_arg_fini().
*/
void starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, ...);

/**
   Structure to be used for starpu_codelet_pack_arg_init() & co, and
   starpu_codelet_unpack_arg_init() & co. The contents is public,
   however users should not directly access it, but only use as a
   parameter to the appropriate functions.
*/
struct starpu_codelet_pack_arg_data
{
	char *arg_buffer;
	size_t arg_buffer_size;
	size_t arg_buffer_used;
	size_t current_offset;
	int nargs;
};

/**
   Initialize struct starpu_codelet_pack_arg before calling
   starpu_codelet_pack_arg() and starpu_codelet_pack_arg_fini(). This
   will simply initialize the content of the structure.
*/
void starpu_codelet_pack_arg_init(struct starpu_codelet_pack_arg_data *state);

/**
   Pack one argument into struct starpu_codelet_pack_arg \p state.
   That structure has to be initialized before with
   starpu_codelet_pack_arg_init(), and after all
   starpu_codelet_pack_arg() calls performed,
   starpu_codelet_pack_arg_fini() has to be used to get the \p cl_arg
   and \p cl_arg_size to be put in the task.
*/
void starpu_codelet_pack_arg(struct starpu_codelet_pack_arg_data *state, const void *ptr, size_t ptr_size);

/**
   Finish packing data, after calling starpu_codelet_pack_arg_init()
   once and starpu_codelet_pack_arg() several times.
*/
void starpu_codelet_pack_arg_fini(struct starpu_codelet_pack_arg_data *state, void **cl_arg, size_t *cl_arg_size);

/**
   Retrieve the arguments of type ::STARPU_VALUE associated to a
   task automatically created using the function starpu_task_insert(). If
   any parameter's value is 0, unpacking will stop there and ignore the remaining
   parameters.
*/
void starpu_codelet_unpack_args(void *cl_arg, ...);

/**
   Initialize \p state with \p cl_arg and \p cl_arg_size. This has to
   be called before calling starpu_codelet_unpack_arg().
*/
void starpu_codelet_unpack_arg_init(struct starpu_codelet_pack_arg_data *state, void *cl_arg, size_t cl_arg_size);

/**
   Unpack the next argument of size \p size from \p state into \p ptr with a copy.
   \p state has to be initialized before with starpu_codelet_unpack_arg_init().
*/
void starpu_codelet_unpack_arg(struct starpu_codelet_pack_arg_data *state, void *ptr, size_t size);

/**
   Unpack the next argument of unknown size from \p state into \p ptr
   with a copy. \p ptr is allocated before copying in it the value of
   the argument.
   The size of the argument is returned in \p size.
   \p has to be initialized before with starpu_codelet_unpack_arg_init().
*/
void starpu_codelet_dup_arg(struct starpu_codelet_pack_arg_data *state, void **ptr, size_t *size);

/**
   Unpack the next argument of unknown size from \p state into \p ptr.
   \p ptr will be a pointer to the memory of the argument.
   The size of the argument is returned in \p size.
   \p has to be initialized before with starpu_codelet_unpack_arg_init().
*/
void starpu_codelet_pick_arg(struct starpu_codelet_pack_arg_data *state, void **ptr, size_t *size);

/**
   Finish unpacking data, after calling starpu_codelet_unpack_arg_init()
   once and starpu_codelet_unpack_arg() or starpu_codelet_dup_arg() or
   starpu_codelet_pick_arg() several times.
*/
void starpu_codelet_unpack_arg_fini(struct starpu_codelet_pack_arg_data *state);

/**
   Call this function during unpacking to skip saving the argument in ptr.
*/
void starpu_codelet_unpack_discard_arg(struct starpu_codelet_pack_arg_data *state);

/**
   Similar to starpu_codelet_unpack_args(), but if any parameter is 0,
   copy the part of \p cl_arg that has not been read in \p buffer
   which can then be used in a later call to one of the unpack
   functions.
*/
void starpu_codelet_unpack_args_and_copyleft(void *cl_arg, void *buffer, size_t buffer_size, ...);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_UTIL_H__ */
