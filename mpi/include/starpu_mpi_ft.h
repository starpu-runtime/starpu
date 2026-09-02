/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Included before the guard, as starpu_task.h does with starpu.h: the
 * declarations below use starpu_mpi_tag_t, and starpu_mpi.h includes this
 * header right after defining it. Including starpu_mpi.h first means this
 * header is processed at that point whichever of the two is included first. */
#include <starpu_mpi.h>

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
   Wrapped function to register a checkpoint template \p cp_template with the given arguments.
   It is then ready to use with starpu_mpi_checkpoint_template_submit() during the program execution.
   This command executes starpu_mpi_checkpoint_template_create(), adds the given checkpoint entry and freezes the
   checkpoint, and therefore can no longer be modified.
   A unique checkpoint id \p cp_id is requested from the user in order to create several templates and to
   match with a corresponding starpu_mpi_init_from_checkpoint().

   The arguments following the \p cp_template and the \p cp_id can be of the following types:
   <ul>
   <li> ::STARPU_R followed by a data handle and the backup rank;
   <li> ::STARPU_DATA_ARRAY followed by an array of data handles,
   its number of elements and a backup rank (non functional);
   <li> ::STARPU_VALUE followed by a pointer to the unregistered value,
   its size in bytes, a unique tag (as the ones given for data handle registering)
   and the function giving the back up rank of the rank argument : int(backup_of)(int) .
   <li> The argument list must be ended by the value 0.
   </ul>
*/
int starpu_mpi_checkpoint_template_register(starpu_mpi_checkpoint_template_t *cp_template, int cp_id, int cp_domain, ...);

/**
   Create a new checkpoint template. A unique checkpoint id \p cp_id is requested from
   the user in order to create several templates and to
   match with a corresponding starpu_mpi_init_from_checkpoint().
   Note a template must be frozen with starpu_mpi_checkpoint_template_freeze() in order to use it
   with starpu_mpi_checkpoint_template_submit().
*/
int starpu_mpi_checkpoint_template_create(starpu_mpi_checkpoint_template_t *cp_template, int cp_id, int cp_domain);

/**
   Add a single entry to a checkpoint template previously created with starpu_mpi_checkpoint_template_create().
   As many entries can be added to a template with as many argument to a single function call, or with as many
   calls to this function.
   Once all the entry added, the
   template must be frozen before using starpu_mpi_checkpoint_template_submit().

   The arguments following the \p cp_template can be of the following types:
   <ul>
   <li> ::STARPU_R followed by a data handle and the backup rank;
   <li> (non functional) ::STARPU_DATA_ARRAY followed by an array of data handles,
   its number of elements and a backup rank (non functional);
   <li> ::STARPU_VALUE followed by a pointer to the unregistered value,
   its size in bytes, a unique tag (as the ones given for data handle registering)
   and the function giving the back up rank of the rank argument : int(backup_of)(int) .
   <li> The argument list must be ended by the value 0.
   </ul>
*/
int starpu_mpi_checkpoint_template_add_entry(starpu_mpi_checkpoint_template_t *cp_template, ...);

/**
   Freeze the given template.
   A frozen template can no longer be modified with starpu_mpi_checkpoint_template_add_entry().
   A template must be frozen before using starpu_mpi_checkpoint_template_submit().
*/
int starpu_mpi_checkpoint_template_freeze(starpu_mpi_checkpoint_template_t *cp_template);

/**
   Submit the checkpoint to StarPU, and can be seen as a cut in the task graph. StarPU will save the data as currently
   described in the submission. Note that the data external to StarPu (::STARPU_VALUE) will be saved with the current value
   at submission time (when starpu_mpi_checkpoint_template_submit() is called).
   The data internal to StarPU (aka handles given with ::STARPU_R) will be saved with their value at
   execution time (when the task submitted before the starpu_mpi_checkpoint_template_submit() have been executed,
   and before this data is modified by the tasks submitted after the starpu_mpi_checkpoint_template_submit())
*/
int starpu_mpi_checkpoint_template_submit(starpu_mpi_checkpoint_template_t cp_template, int prio);

int starpu_mpi_checkpoint_template_print(starpu_mpi_checkpoint_template_t cp_template);

/**
   Configure the directory \p path, on a filesystem shared by all the
   nodes, to be used as the persistent checkpoint storage backend. The
   directory is created if it does not exist. This must be called before
   starpu_mpi_checkpoint_flush_to_storage().
   Return 0 on success, -1 on error.
   See \ref MPICheckpointPersistent for more details.
*/
int starpu_mpi_checkpoint_set_storage_path(const char *path);

/**
   Write the entries of the checkpoint template \p cp_template owned by the
   calling node to the storage directory, so that a later run can resume
   from them. All the nodes must call this function.

   Each call writes a new checkpoint, and either commits it completely or
   removes it and keeps the previous one, so a checkpoint found in the
   storage is never partial.

   This function does \b not wait for the submitted tasks to complete. It
   waits only for the data it reads, so a checkpoint can commit without
   waiting for the work in flight. Call starpu_task_wait_for_all() before it
   for a quiescent checkpoint.

   Return 0 on success, -1 if no storage directory has been configured or
   if the checkpoint was discarded.
   See \ref MPICheckpointPersistent for more details.
*/
int starpu_mpi_checkpoint_flush_to_storage(starpu_mpi_checkpoint_template_t cp_template);

/**
   Look in the storage directory \p storage_path for a checkpoint left by a
   previous run, and set \p restart_flag to 1 if one is found, 0 otherwise.
   The checkpoint is then described by
   starpu_mpi_checkpoint_get_restart_cp_id(),
   starpu_mpi_checkpoint_get_restart_cp_inst() and
   starpu_mpi_checkpoint_get_restart_n_ranks().

   Call this early, before registering any template. It also configures the
   storage directory, so it replaces
   starpu_mpi_checkpoint_set_storage_path().

   Return 0 on success, -1 on error.
   See \ref MPICheckpointRestart for more details.
*/
int starpu_mpi_init_from_checkpoint(const char *storage_path, int *restart_flag);

/**
   Return the checkpoint id recorded by
   starpu_mpi_init_from_checkpoint(), or -1 if no checkpoint was found.
*/
int starpu_mpi_checkpoint_get_restart_cp_id(void);

/**
   Return the checkpoint instance recorded by
   starpu_mpi_init_from_checkpoint(), or -1 if no checkpoint was found.
*/
int starpu_mpi_checkpoint_get_restart_cp_inst(void);

/**
   Return the number of nodes which produced the checkpoint recorded by
   starpu_mpi_init_from_checkpoint(), or -1 if no checkpoint was found.
   The application can compare it with the current number of nodes to
   detect that it is restarting with a different number of nodes, and
   remap the checkpoint entries accordingly.
   See \ref MPICheckpointRestart for more details.
*/
int starpu_mpi_checkpoint_get_restart_n_ranks(void);

/**
   Restore \p handle from the checkpoint \p cp_id / \p cp_inst, using the
   entry that the node of rank \p old_rank wrote under the message tag
   \p tag.

   Since \p old_rank is a parameter, a run can resume on a different number
   of nodes: the application decides which old ranks each node takes over,
   and restores their entries.

   Return 0 on success, -1 if the entry does not exist, does not match the
   size of \p handle, or could not be read.
   See \ref MPICheckpointRestart for more details.
*/
int starpu_mpi_checkpoint_restore_handle(int cp_id, int cp_inst, int old_rank, starpu_mpi_tag_t tag, starpu_data_handle_t handle);

/**
   Similar to starpu_mpi_checkpoint_restore_handle(), but restore an
   entry registered as data external to StarPU (::STARPU_VALUE) into the
   buffer \p ptr. The value of \p size must match the size the entry had
   when the checkpoint was written.

   Return 0 on success, -1 if the checkpoint entry does not exist, has a
   different size, or could not be read.
*/
int starpu_mpi_checkpoint_restore_value(int cp_id, int cp_inst, int old_rank, starpu_mpi_tag_t tag, void *ptr, size_t size);

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
static inline int starpu_mpi_checkpoint_set_storage_path(const char *path STARPU_ATTRIBUTE_UNUSED) { return 0; }
static inline int starpu_mpi_checkpoint_flush_to_storage(starpu_mpi_checkpoint_template_t cp_template STARPU_ATTRIBUTE_UNUSED) { return 0; }
static inline int starpu_mpi_init_from_checkpoint(const char *storage_path STARPU_ATTRIBUTE_UNUSED, int *restart_flag) { *restart_flag = 0; return 0; }
static inline int starpu_mpi_checkpoint_get_restart_cp_id(void) { return -1; }
static inline int starpu_mpi_checkpoint_get_restart_cp_inst(void) { return -1; }
static inline int starpu_mpi_checkpoint_get_restart_n_ranks(void) { return -1; }
static inline int starpu_mpi_checkpoint_restore_handle(int cp_id STARPU_ATTRIBUTE_UNUSED, int cp_inst STARPU_ATTRIBUTE_UNUSED, int old_rank STARPU_ATTRIBUTE_UNUSED, starpu_mpi_tag_t tag STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED) { return -1; }
static inline int starpu_mpi_checkpoint_restore_value(int cp_id STARPU_ATTRIBUTE_UNUSED, int cp_inst STARPU_ATTRIBUTE_UNUSED, int old_rank STARPU_ATTRIBUTE_UNUSED, starpu_mpi_tag_t tag STARPU_ATTRIBUTE_UNUSED, void *ptr STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED) { return -1; }

#endif // STARPU_USE_MPI_FT

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_MPI_FT_H__ */
