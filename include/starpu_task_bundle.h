/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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

#ifndef __STARPU_TASK_BUNDLE_H__
#define __STARPU_TASK_BUNDLE_H__

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Task_Bundles Task Bundles
   @{
*/

struct starpu_task;
struct starpu_perfmodel_arch;

/**
   Opaque structure describing a list of tasks that should be
   scheduled on the same worker whenever it’s possible. It must be
   considered as a hint given to the scheduler as there is no
   guarantee that they will be executed on the same worker.
*/
typedef struct _starpu_task_bundle *starpu_task_bundle_t;

/**
   Factory function creating and initializing \p bundle, when the call
   returns, memory needed is allocated and \p bundle is ready to use.
*/
void starpu_task_bundle_create(starpu_task_bundle_t *bundle);

/**
   Insert \p task in \p bundle. Until \p task is removed from \p
   bundle its expected length and data transfer time will be
   considered along those of the other tasks of bundle. This function
   must not be called if \p bundle is already closed and/or \p task is
   already submitted. On success, it returns 0. There are two cases of
   error : if \p bundle is already closed it returns <c>-EPERM</c>, if
   \p task was already submitted it returns <c>-EINVAL</c>.
*/
int starpu_task_bundle_insert(starpu_task_bundle_t bundle, struct starpu_task *task);

/**
   Remove \p task from \p bundle. Of course \p task must have been
   previously inserted in \p bundle. This function must not be called
   if \p bundle is already closed and/or \p task is already submitted.
   Doing so would result in undefined behaviour. On success, it
   returns 0. If \p bundle is already closed it returns
   <c>-ENOENT</c>.
*/
int starpu_task_bundle_remove(starpu_task_bundle_t bundle, struct starpu_task *task);

/**
   Inform the runtime that the user will not modify \p bundle anymore,
   it means no more inserting or removing task. Thus the runtime can
   destroy it when possible.
*/
void starpu_task_bundle_close(starpu_task_bundle_t bundle);

/**
   Return the expected duration of \p bundle in micro-seconds.
*/
double starpu_task_bundle_expected_length(starpu_task_bundle_t bundle, struct starpu_perfmodel_arch *arch, unsigned nimpl);

/**
   Return the time (in micro-seconds) expected to transfer all data used within \p bundle.
*/
double starpu_task_bundle_expected_data_transfer_time(starpu_task_bundle_t bundle, unsigned memory_node);

/**
   Return the expected energy consumption of \p bundle in J.
*/
double starpu_task_bundle_expected_energy(starpu_task_bundle_t bundle, struct starpu_perfmodel_arch *arch, unsigned nimpl);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_BUNDLE_H__ */
