/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2016       Uppsala University
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

#ifndef __STARPU_TASK_DEP_H__
#define __STARPU_TASK_DEP_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Explicit_Dependencies Explicit Dependencies
   @{
*/

/**
   Declare task dependencies between a \p task and an array of tasks
   of length \p ndeps. This function must be called prior to the
   submission of the task, but it may called after the submission or
   the execution of the tasks in the array, provided the tasks are
   still valid (i.e. they were not automatically destroyed). Calling
   this function on a task that was already submitted or with an entry
   of \p task_array that is no longer a valid task results in an
   undefined behaviour. If \p ndeps is 0, no dependency is added. It
   is possible to call starpu_task_declare_deps_array() several times
   on the same task, in this case, the dependencies are added. It is
   possible to have redundancy in the task dependencies.
*/
void starpu_task_declare_deps_array(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);

/**
   Declare task dependencies between a \p task and an series of \p
   ndeps tasks, similarly to starpu_task_declare_deps_array(), but the
   tasks are passed after \p ndeps, which indicates how many tasks \p
   task shall be made to depend on. If \p ndeps is 0, no dependency is
   added.
*/
void starpu_task_declare_deps(struct starpu_task *task, unsigned ndeps, ...);

/**
   Declare task end dependencies between a \p task and an array of
   tasks of length \p ndeps. \p task will appear as terminated not
   only when \p task is termination, but also when the tasks of \p
   task_array have terminated. This function must be called prior to
   the termination of the task, but it may called after the submission
   or the execution of the tasks in the array, provided the tasks are
   still valid (i.e. they were not automatically destroyed). Calling
   this function on a task that was already terminated or with an
   entry of \p task_array that is no longer a valid task results in an
   undefined behaviour. If \p ndeps is 0, no dependency is added. It
   is possible to call starpu_task_declare_end_deps_array() several
   times on the same task, in this case, the dependencies are added.
   It is currently not implemented to have redundancy in the task
   dependencies.
*/
void starpu_task_declare_end_deps_array(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);

/**
   Declare task end dependencies between a \p task and an series of \p
   ndeps tasks, similarly to starpu_task_declare_end_deps_array(), but
   the tasks are passed after \p ndeps, which indicates how many tasks
   \p task 's termination shall be made to depend on. If \p ndeps is
   0, no dependency is added.
*/
void starpu_task_declare_end_deps(struct starpu_task *task, unsigned ndeps, ...);

/**
   Fill \p task_array with the list of tasks which are direct children
   of \p task. \p ndeps is the size of \p task_array.  This function
   returns the number of direct children. \p task_array can be set to
   <c>NULL</c> if \p ndeps is 0, which allows to compute the number of
   children before allocating an array to store them. This function
   can only be called if \p task has not completed yet, otherwise the
   results are undefined. The result may also be outdated if some
   additional dependency has been added in the meanwhile.
*/
int starpu_task_get_task_succs(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);

/**
   Behave like starpu_task_get_task_succs(), except that it only
   reports tasks which will go through the scheduler, thus avoiding
   tasks with not codelet, or with explicit placement.
*/
int starpu_task_get_task_scheduled_succs(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);

/**
   Add \p nb_deps end dependencies to the task \p t. This means the
   task will not terminate until the required number of calls to the
   function starpu_task_end_dep_release() has been made.
*/
void starpu_task_end_dep_add(struct starpu_task *t, int nb_deps);

/**
   Unlock 1 end dependency to the task \p t. This function must be
   called after starpu_task_end_dep_add().
*/
void starpu_task_end_dep_release(struct starpu_task *t);

/**
   Define a task logical identifer. It is possible to associate a task
   with a unique <em>tag</em> chosen by the application, and to
   express dependencies between tasks by the means of those tags. To
   do so, fill the field starpu_task::tag_id with a tag number (can be
   arbitrary) and set the field starpu_task::use_tag to 1. If
   starpu_tag_declare_deps() is called with this tag number, the task
   will not be started until the tasks which holds the declared
   dependency tags are completed.
*/
typedef uint64_t starpu_tag_t;

/**
   Specify the dependencies of the task identified by tag \p id. The
   first argument specifies the tag which is configured, the second
   argument gives the number of tag(s) on which \p id depends. The
   following arguments are the tags which have to be terminated to
   unlock the task. This function must be called before the associated
   task is submitted to StarPU with starpu_task_submit().

   <b>WARNING! Use with caution</b>. Because of the variable arity of
   starpu_tag_declare_deps(), note that the last arguments must be of
   type ::starpu_tag_t : constant values typically need to be
   explicitly casted. Otherwise, due to integer sizes and argument
   passing on the stack, the C compiler might consider the tag
   <c>0x200000003</c> instead of <c>0x2</c> and <c>0x3</c> when
   calling <c>starpu_tag_declare_deps(0x1, 2, 0x2, 0x3)</c>. Using the
   starpu_tag_declare_deps_array() function avoids this hazard.

   \code{.c}
   //  Tag 0x1 depends on tags 0x32 and 0x52
   starpu_tag_declare_deps((starpu_tag_t)0x1, 2, (starpu_tag_t)0x32, (starpu_tag_t)0x52);
   \endcode
*/
void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...);

/**
   Similar to starpu_tag_declare_deps(), except that its does not take
   a variable number of arguments but an \p array of tags of size \p
   ndeps.

   \code{.c}
   // Tag 0x1 depends on tags 0x32 and 0x52
   starpu_tag_t tag_array[2] = {0x32, 0x52};
   starpu_tag_declare_deps_array((starpu_tag_t)0x1, 2, tag_array);
   \endcode
*/
void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array);

/**
   Block until the task associated to tag \p id has been executed.
   This is a blocking call which must therefore not be called within
   tasks or callbacks, but only from the application directly. It is
   possible to synchronize with the same tag multiple times, as long
   as the starpu_tag_remove() function is not called. Note that it is
   still possible to synchronize with a tag associated to a task for
   which the strucuture starpu_task was freed (e.g. if the field
   starpu_task::destroy was enabled).
*/
int starpu_tag_wait(starpu_tag_t id);

/**
   Similar to starpu_tag_wait() except that it blocks until all the \p
   ntags tags contained in the array \p id are terminated.
*/
int starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id);

/**
   Clear the <em>already notified</em> status of a tag which is not
   associated with a task. Before that, calling
   starpu_tag_notify_from_apps() again will not notify the successors.
   After that, the next call to starpu_tag_notify_from_apps() will
   notify the successors.
*/
void starpu_tag_restart(starpu_tag_t id);

/**
   Release the resources associated to tag \p id. It can be called
   once the corresponding task has been executed and when there is no
   other tag that depend on this tag anymore.
*/
void starpu_tag_remove(starpu_tag_t id);

/**
   Explicitly unlock tag \p id. It may be useful in the case of
   applications which execute part of their computation outside StarPU
   tasks (e.g. third-party libraries). It is also provided as a
   convenient tool for the programmer, for instance to entirely
   construct the task DAG before actually giving StarPU the
   opportunity to execute the tasks. When called several times on the
   same tag, notification will be done only on first call, thus
   implementing "OR" dependencies, until the tag is restarted using
   starpu_tag_restart().
*/
void starpu_tag_notify_from_apps(starpu_tag_t id);

/**
   Atomically call starpu_tag_notify_from_apps() and starpu_tag_restart() on tag
   \p id.
   This is useful with cyclic graphs, when we want to safely trigger its startup.
*/
void starpu_tag_notify_restart_from_apps(starpu_tag_t id);

struct starpu_task *starpu_tag_get_task(starpu_tag_t id);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_DEP_H__ */
