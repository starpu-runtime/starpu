/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPURM_H
#define __STARPURM_H
#include <hwloc.h>
#include <starpurm_config.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Interop_Support Interoperability Support
   @brief This section describes the interface supplied by StarPU to
   interoperate with other runtime systems.
   @{
*/

/**
   StarPU Resource Manager return type.
*/
enum e_starpurm_drs_ret
{
	/**
	   Dynamic resource sharing operation succeeded.
	*/
	starpurm_DRS_SUCCESS = 0,

	/**
	   Dynamic resource sharing is disabled.
	*/
	starpurm_DRS_DISABLD = -1,
	/**
	   Dynamic resource sharing operation is not authorized or
	   implemented.
	*/
	starpurm_DRS_PERM    = -2,
	/**
	   Dynamic resource sharing operation has been called with one
	   or more invalid parameters.
	*/
	starpurm_DRS_EINVAL  = -3
#if 0
	/* Unused for now */
	starpurm_DRS_NOTED,
	starpurm_DRS_REQST
#endif
};
typedef int starpurm_drs_ret_t;
typedef void *starpurm_drs_desc_t;
typedef void *starpurm_drs_cbs_t;
typedef void (*starpurm_drs_cb_t)(void *);
typedef void *starpurm_block_cond_t;
typedef int (*starpurm_polling_t)(void *);

/**
   @name Initialisation
   @{
*/

/**
   Resource enforcement
*/
void starpurm_initialize_with_cpuset(hwloc_cpuset_t initially_owned_cpuset);

/**
   Initialize StarPU and the StarPU-RM resource management module. The
   starpu_init() function should not have been called before the call
   to starpurm_initialize(). The starpurm_initialize() function will
   take care of this
*/
void starpurm_initialize(void);

/**
   Shutdown StarPU-RM and StarPU. The starpu_shutdown() function
   should not be called before. The starpurm_shutdown() function will
   take care of this.
*/
void starpurm_shutdown(void);

/** @} */

/**
   @name Spawn
   @{
*/

/**
   Allocate a temporary context spanning the units selected in the
   cpuset bitmap, set it as the default context for the current
   thread, and call user function \p f. Upon the return of user
   function \p f, the temporary context is freed and the previous
   default context for the current thread is restored.
*/
void starpurm_spawn_kernel_on_cpus(void *data, void(*f)(void *), void *args, hwloc_cpuset_t cpuset);

/**
   Spawn a POSIX thread and returns immediately. The thread spawned
   will allocate a temporary context spanning the units selected in
   the cpuset bitmap, set it as the default context for the current
   thread, and call user function \p f. Upon the return of user
   function \p f, the temporary context will be freed and the previous
   default context for the current thread restored. A user specified
   callback \p cb_f will be called just before the termination of the
   thread.
*/
void starpurm_spawn_kernel_on_cpus_callback(void *data, void(*f)(void *), void *args, hwloc_cpuset_t cpuset, void(*cb_f)(void *), void *cb_args);

void starpurm_spawn_kernel_callback(void *data, void(*f)(void *), void *args, void(*cb_f)(void *), void *cb_args);

/** @} */

/**
   @name DynamicResourceSharing
   @{
*/

/**
   Turn-on dynamic resource sharing support.
*/
starpurm_drs_ret_t starpurm_set_drs_enable(starpurm_drs_desc_t *spd);

/**
   Turn-off dynamic resource sharing support.
*/
starpurm_drs_ret_t starpurm_set_drs_disable(starpurm_drs_desc_t *spd);

/**
   Return the state of the dynamic resource sharing support (\p =!0
   enabled, \p =0 disabled).
*/
int starpurm_drs_enabled_p(void);

/**
   Set the maximum number of CPU computing units available for StarPU
   computations to \p max. This number cannot exceed the maximum
   number of StarPU's CPU worker allocated at start-up time.
*/
starpurm_drs_ret_t starpurm_set_max_parallelism(starpurm_drs_desc_t *spd, int max);

#if 0
/* Unused for now */
starpurm_drs_ret_t starpurm_callback_set(starpurm_drs_desc_t *spd, starpurm_drs_cbs_t which, starpurm_drs_cb_t callback);
starpurm_drs_ret_t starpurm_callback_get(starpurm_drs_desc_t *spd, starpurm_drs_cbs_t which, starpurm_drs_cb_t *callback);
#endif

/**
   Extend StarPU's default scheduling context to execute tasks on
   worker corresponding to logical unit \p cpuid. If StarPU does not
   have a worker thread initialized for logical unit \p cpuid, do
   nothing.
*/
starpurm_drs_ret_t starpurm_assign_cpu_to_starpu(starpurm_drs_desc_t *spd, int cpuid);

/**
   Extend StarPU's default scheduling context to execute tasks on \p
   ncpus more workers, up to the number of StarPU worker threads
   initialized.
*/
starpurm_drs_ret_t starpurm_assign_cpus_to_starpu(starpurm_drs_desc_t *spd, int ncpus);

/**
   Extend StarPU's default scheduling context to execute tasks on the
   additional logical units selected in \p mask. Logical units of \p
   mask for which no StarPU worker is initialized are silently ignored.
*/
starpurm_drs_ret_t starpurm_assign_cpu_mask_to_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Set StarPU's default scheduling context to execute tasks on all
   available logical units for which a StarPU worker has been
   initialized.
*/
starpurm_drs_ret_t starpurm_assign_all_cpus_to_starpu(starpurm_drs_desc_t *spd);

/**
   Shrink StarPU's default scheduling context so as to not execute
   tasks on worker corresponding to logical unit \p cpuid. If StarPU
   does not have a worker thread initialized for logical unit \p
   cpuid, do nothing.
*/
starpurm_drs_ret_t starpurm_withdraw_cpu_from_starpu(starpurm_drs_desc_t *spd, int cpuid);

/**
   Shrink StarPU's default scheduling context to execute tasks on \p
   ncpus less workers.
*/
starpurm_drs_ret_t starpurm_withdraw_cpus_from_starpu(starpurm_drs_desc_t *spd, int ncpus);

/**
   Shrink StarPU's default scheduling context so as to not execute
   tasks on the logical units selected in \p mask. Logical units of \p
   mask for which no StarPU worker is initialized are silently ignored.
*/
starpurm_drs_ret_t starpurm_withdraw_cpu_mask_from_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Shrink StarPU's default scheduling context so as to remove all
   logical units.
*/
starpurm_drs_ret_t starpurm_withdraw_all_cpus_from_starpu(starpurm_drs_desc_t *spd);

/* --- */

/**
   Synonym for starpurm_assign_all_cpus_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend(starpurm_drs_desc_t *spd);

/**
   Synonym for starpurm_assign_cpu_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend_cpu(starpurm_drs_desc_t *spd, int cpuid);

/**
   Synonym for starpurm_assign_cpus_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend_cpus(starpurm_drs_desc_t *spd, int ncpus);

/**
   Synonym for starpurm_assign_cpu_mask_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Synonym for starpurm_withdraw_all_cpus_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim(starpurm_drs_desc_t *spd);

/**
   Synonym for starpurm_withdraw_cpu_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim_cpu(starpurm_drs_desc_t *spd, int cpuid);

/**
   Synonym for starpurm_withdraw_cpus_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim_cpus(starpurm_drs_desc_t *spd, int ncpus);

/**
   Synonym for starpurm_withdraw_cpu_mask_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Synonym for starpurm_withdraw_all_cpus_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire(starpurm_drs_desc_t *spd);

/**
   Synonym for starpurm_withdraw_cpu_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire_cpu(starpurm_drs_desc_t *spd, int cpuid);

/**
   Synonym for starpurm_withdraw_cpus_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire_cpus(starpurm_drs_desc_t *spd, int ncpus);

/**
   Synonym for starpurm_withdraw_cpu_mask_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Synonym for starpurm_assign_all_cpus_to_starpu().
*/
starpurm_drs_ret_t starpurm_return_all(starpurm_drs_desc_t *spd);

/**
   Synonym for starpurm_assign_cpu_to_starpu().
*/
starpurm_drs_ret_t starpurm_return_cpu(starpurm_drs_desc_t *spd, int cpuid);

#if 0
/* Pause/resume (not implemented) */
starpurm_drs_ret_t starpurm_create_block_condition(starpurm_block_cond_t *cond);
void starpurm_block_current_task(starpurm_block_cond_t *cond);
void starpurm_signal_block_condition(starpurm_block_cond_t *cond);

void starpurm_register_polling_service(const char *service_name, starpurm_polling_t function, void *data);
void starpurm_unregister_polling_service(const char *service_name, starpurm_polling_t function, void *data);
#endif

/** @} */

/**
   @name Devices
   @{
*/

/**
   Return the device type ID constant associated to the device type name.
   Valid names for \p type_str are:
   - \c "cpu": regular CPU unit;
   - \c "opencl": OpenCL device unit;
   - \c "cuda": nVidia CUDA device unit;
   - \c "mic": Intel KNC type device unit.
*/
int starpurm_get_device_type_id(const char *type_str);

/**
   Return the device type name associated to the device type ID
   constant.
*/
const char *starpurm_get_device_type_name(int type_id);

/**
   Return the number of initialized StarPU worker for the device type
   \p type_id.
*/
int starpurm_get_nb_devices_by_type(int type_id);

/**
   Return the unique ID assigned to the \p device_rank nth device of
   type \p type_id.
*/
int starpurm_get_device_id(int type_id, int device_rank);

/**
   Extend StarPU's default scheduling context to use \p unit_rank nth
   device of type \p type_id.
*/
starpurm_drs_ret_t starpurm_assign_device_to_starpu(starpurm_drs_desc_t *spd, int type_id, int unit_rank);

/**
   Extend StarPU's default scheduling context to use \p ndevices more
   devices of type \p type_id, up to the number of StarPU workers
   initialized for such device type.
 */
starpurm_drs_ret_t starpurm_assign_devices_to_starpu(starpurm_drs_desc_t *spd, int type_id, int ndevices);

/**
   Extend StarPU's default scheduling context to use additional
   devices as designated by their corresponding StarPU worker
   thread(s) CPU-set \p mask.
 */
starpurm_drs_ret_t starpurm_assign_device_mask_to_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Extend StarPU's default scheduling context to use all devices of
   type \p type_id for which it has a worker thread initialized.
*/
starpurm_drs_ret_t starpurm_assign_all_devices_to_starpu(starpurm_drs_desc_t *spd, int type_id);

/**
   Shrink StarPU's default scheduling context to not use \p unit_rank
   nth device of type \p type_id.
 */
starpurm_drs_ret_t starpurm_withdraw_device_from_starpu(starpurm_drs_desc_t *spd, int type_id, int unit_rank);

/**
   Shrink StarPU's default scheduling context to use \p ndevices less
   devices of type \p type_id.
*/
starpurm_drs_ret_t starpurm_withdraw_devices_from_starpu(starpurm_drs_desc_t *spd, int type_id, int ndevices);

/**
   Shrink StarPU's default scheduling context to not use devices
   designated by their corresponding StarPU worker thread(s) CPU-set
   \p mask.
*/
starpurm_drs_ret_t starpurm_withdraw_device_mask_from_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Shrink StarPU's default scheduling context to use no devices of
   type \p type_id.
*/
starpurm_drs_ret_t starpurm_withdraw_all_devices_from_starpu(starpurm_drs_desc_t *spd, int type_id);

/* --- */

/**
   Synonym for starpurm_assign_device_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);

/**
   Synonym for starpurm_assign_devices_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices);

/**
   Synonym for starpurm_assign_device_mask_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Synonym for starpurm_assign_all_devices_to_starpu().
*/
starpurm_drs_ret_t starpurm_lend_all_devices(starpurm_drs_desc_t *spd, int type_id);

/**
   Synonym for starpurm_withdraw_device_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);

/**
   Synonym for starpurm_withdraw_devices_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices);

/**
   Synonym for starpurm_withdraw_device_mask_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Synonym for starpurm_withdraw_all_devices_from_starpu().
*/
starpurm_drs_ret_t starpurm_reclaim_all_devices(starpurm_drs_desc_t *spd, int type_id);

/**
   Synonym for starpurm_withdraw_device_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);

/**
   Synonym for starpurm_withdraw_devices_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices);

/**
   Synonym for starpurm_withdraw_device_mask_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

/**
   Synonym for starpurm_withdraw_all_devices_from_starpu().
*/
starpurm_drs_ret_t starpurm_acquire_all_devices(starpurm_drs_desc_t *spd, int type_id);

/**
   Synonym for starpurm_assign_all_devices_to_starpu().
*/
starpurm_drs_ret_t starpurm_return_all_devices(starpurm_drs_desc_t *spd, int type_id);

/**
   Synonym for starpurm_assign_device_to_starpu().
*/
starpurm_drs_ret_t starpurm_return_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);

/** @} */

/**
   @name CpusetsQueries
   @{
*/

/**
   Return the CPU-set of the StarPU worker associated to the \p
   unit_rank nth unit of type \p type_id.
*/
hwloc_cpuset_t starpurm_get_device_worker_cpuset(int type_id, int unit_rank);

/**
   Return the cumulated CPU-set of all StarPU worker threads.
*/
hwloc_cpuset_t starpurm_get_global_cpuset(void);

/**
   Return the CPU-set of the StarPU worker threads currently selected
   in the default StarPU's scheduling context.
 */
hwloc_cpuset_t starpurm_get_selected_cpuset(void);

/**
   Return the cumulated CPU-set of all CPU StarPU worker threads.
*/
hwloc_cpuset_t starpurm_get_all_cpu_workers_cpuset(void);

/**
   Return the cumulated CPU-set of all "non-CPU" StarPU worker
   threads.
 */
hwloc_cpuset_t starpurm_get_all_device_workers_cpuset(void);

/**
   Return the cumulated CPU-set of all StarPU worker threads for
   devices of type \p typeid.
*/
hwloc_cpuset_t starpurm_get_all_device_workers_cpuset_by_type(int typeid);

/** @} */
/** @} */

#ifdef __cplusplus
}
#endif
#endif /* __STARPURM_H */
