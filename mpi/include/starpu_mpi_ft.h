/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_FT_H__
#define __STARPU_MPI_FT_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

struct _starpu_mpi_checkpoint_template;
typedef struct _starpu_mpi_checkpoint_template *starpu_mpi_checkpoint_template_t;

#if defined(STARPU_USE_MPI_FT)
/**
   @defgroup API_MPI_FT_Support MPI Fault Tolerance Support
   @{
*/

/**
   Initialise the checkpoint mechanism
*/
int starpu_mpi_checkpoint_init(void);

/**
   Shutdown the checkpoint mechanism
*/
int starpu_mpi_checkpoint_shutdown(void);

/**
 * Wrapped function to register a checkpoint template \p cp_template with the given arguments.
 * It is then ready to use with ::starpu_mpi_checkpoint_template_submit() during the program execution.
 * This command executes ::starpu_mpi_checkpoint_template_create(), adds the given checkpoint entry and freezes the
 * checkpoint, and therefore can no longer be modified.
 * A unique checkpoint id \p cp_id is requested from the user in order to create several templates and to
 * match with a corresponding ::starpu_mpi_init_from_checkpoint() (not implemented yet).
 *
 * The arguments following the \p cp_template and the \p cp_id can be of the following types:
 * <ul>
 * <li> ::STARPU_R followed by a data handle and the backup rank;
 * <li> ::STARPU_DATA_ARRAY followed by an array of data handles,
 * its number of elements and a backup rank (non functional);
 * <li> ::STARPU_VALUE followed by a pointer to the unregistered value,
 * its size in bytes, a unique tag (as the ones given for data handle registering)
 * and the function giving the back up rank of the rank argument : int(backup_of)(int) .
 * <li> The argument list must be ended by the value 0.
 * </ul>
 */
int starpu_mpi_checkpoint_template_register(starpu_mpi_checkpoint_template_t *cp_template, int cp_id, int cp_domain, ...);

/**
 * Create a new checkpoint template. A unique checkpoint id \p cp_id is requested from
 * the user in order to create several templates and to
 * match with a corresponding ::starpu_mpi_init_from_checkpoint() (not implemented yet).
 * Note a template must be frozen with ::starpu_mpi_checkpoint_template_freeze() in order to use it
 * with ::starpu_mpi_checkpoint_template_submit().
*/
int starpu_mpi_checkpoint_template_create(starpu_mpi_checkpoint_template_t *cp_template, int cp_id, int cp_domain);

/**
 * Add a single entry to a checkpoint template previously created with ::starpu_mpi_checkpoint_template_create().
 * As many entries can be added to a template with as many argument to a single function call, or with as many
 * calls to this function.
 * Once all the entry added, the
 * template must be frozen before using ::starpu_mpi_checkpoint_template_submit().
 *
 * The arguments following the \p cp_template can be of the following types:
 * <ul>
 * <li> ::STARPU_R followed by a data handle and the backup rank;
 * <li> (non functional) ::STARPU_DATA_ARRAY followed by an array of data handles,
 * its number of elements and a backup rank (non functional);
 * <li> ::STARPU_VALUE followed by a pointer to the unregistered value,
 * its size in bytes, a unique tag (as the ones given for data handle registering)
 * and the function giving the back up rank of the rank argument : int(backup_of)(int) .
 * <li> The argument list must be ended by the value 0.
 * </ul>
 */
int starpu_mpi_checkpoint_template_add_entry(starpu_mpi_checkpoint_template_t *cp_template, ...);

/**
 * Freeze the given template.
 * A frozen template can no longer be modified with ::starpu_mpi_checkpoint_template_add_entry().
 * A template must be frozen before using ::starpu_mpi_checkpoint_template_submit().
 */
int starpu_mpi_checkpoint_template_freeze(starpu_mpi_checkpoint_template_t *cp_template);

/**
 * Submit the checkpoint to StarPU, and can be seen as a cut in the task graph. StarPU will save the data as currently
 * described in the submission. Note that the data external to StarPu (::STARPU_VALUE) will be saved with the current value
 * at submission time (when ::starpu_mpi_checkpoint_template_submit() is called).
 * The data internal to StarPU (aka handles given with ::STARPU_R) will be saved with their value at
 * execution time (when the task submitted before the ::starpu_mpi_checkpoint_template_submit() have been executed,
 * and before this data is modified by the tasks submitted after the ::starpu_mpi_checkpoint_template_submit())
 */
int starpu_mpi_checkpoint_template_submit(starpu_mpi_checkpoint_template_t cp_template, int prio);

int starpu_mpi_checkpoint_template_print(starpu_mpi_checkpoint_template_t cp_template);

#else // !STARPU_USE_MPI_FT
static inline int starpu_mpi_checkpoint_template_register(starpu_mpi_checkpoint_template_t *cp_template STARPU_ATTRIBUTE_UNUSED, int cp_id STARPU_ATTRIBUTE_UNUSED, int cp_domain STARPU_ATTRIBUTE_UNUSED, ...) { return 0; }
static inline int starpu_mpi_checkpoint_template_create(starpu_mpi_checkpoint_template_t *cp_template STARPU_ATTRIBUTE_UNUSED, int cp_id STARPU_ATTRIBUTE_UNUSED, int cp_domain STARPU_ATTRIBUTE_UNUSED) { return 0; }
static inline int starpu_mpi_checkpoint_template_add_entry(starpu_mpi_checkpoint_template_t *cp_template STARPU_ATTRIBUTE_UNUSED, ...) { return 0; }
static inline int starpu_mpi_checkpoint_template_freeze(starpu_mpi_checkpoint_template_t *cp_template STARPU_ATTRIBUTE_UNUSED) { return 0; }
static inline int starpu_mpi_checkpoint_template_submit(starpu_mpi_checkpoint_template_t cp_template STARPU_ATTRIBUTE_UNUSED, int prio STARPU_ATTRIBUTE_UNUSED) { return 0; }
static inline int starpu_mpi_ft_turn_on(void) { return 0; }
static inline int starpu_mpi_ft_turn_off(void) { return 0; }
static inline int starpu_mpi_checkpoint_template_print(starpu_mpi_checkpoint_template_t cp_template STARPU_ATTRIBUTE_UNUSED) { return 0; }
static inline int starpu_mpi_checkpoint_init(void) { return 0; }
static inline int starpu_mpi_checkpoint_shutdown(void) { return 0; }

/** @} */

#endif // STARPU_USE_MPI_FT

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_FT_H__
