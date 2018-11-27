/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017,2018                                Inria
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

/* type mapping */
enum e_starpurm_drs_ret
{
	starpurm_DRS_SUCCESS = 0,

	starpurm_DRS_DISABLD = -1,
	starpurm_DRS_PERM    = -2,
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

/* Resource enforcement */
void starpurm_initialize_with_cpuset(hwloc_cpuset_t initially_owned_cpuset);
void starpurm_initialize(void);

void starpurm_shutdown(void);

void starpurm_spawn_kernel_on_cpus(void *data, void(*f)(void *), void *args, hwloc_cpuset_t cpuset);
void starpurm_spawn_kernel_on_cpus_callback(void *data, void(*f)(void *), void *args, hwloc_cpuset_t cpuset, void(*cb_f)(void *), void *cb_args);
void starpurm_spawn_kernel_callback(void *data, void(*f)(void *), void *args, void(*cb_f)(void *), void *cb_args);

/* Dynamic resource sharing */
starpurm_drs_ret_t starpurm_set_drs_enable(starpurm_drs_desc_t *spd);
starpurm_drs_ret_t starpurm_set_drs_disable(starpurm_drs_desc_t *spd);
int starpurm_drs_enabled_p(void);

starpurm_drs_ret_t starpurm_set_max_parallelism(starpurm_drs_desc_t *spd, int max);

#if 0
/* Unused for now */
starpurm_drs_ret_t starpurm_callback_set(starpurm_drs_desc_t *spd, starpurm_drs_cbs_t which, starpurm_drs_cb_t callback);
starpurm_drs_ret_t starpurm_callback_get(starpurm_drs_desc_t *spd, starpurm_drs_cbs_t which, starpurm_drs_cb_t *callback);
#endif

starpurm_drs_ret_t starpurm_assign_cpu_to_starpu(starpurm_drs_desc_t *spd, int cpuid);
starpurm_drs_ret_t starpurm_assign_cpus_to_starpu(starpurm_drs_desc_t *spd, int ncpus);
starpurm_drs_ret_t starpurm_assign_cpu_mask_to_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);
starpurm_drs_ret_t starpurm_assign_all_cpus_to_starpu(starpurm_drs_desc_t *spd);

starpurm_drs_ret_t starpurm_withdraw_cpu_from_starpu(starpurm_drs_desc_t *spd, int cpuid);
starpurm_drs_ret_t starpurm_withdraw_cpus_from_starpu(starpurm_drs_desc_t *spd, int ncpus);
starpurm_drs_ret_t starpurm_withdraw_cpu_mask_from_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);
starpurm_drs_ret_t starpurm_withdraw_all_cpus_from_starpu(starpurm_drs_desc_t *spd);

/* --- */

starpurm_drs_ret_t starpurm_lend(starpurm_drs_desc_t *spd);
starpurm_drs_ret_t starpurm_lend_cpu(starpurm_drs_desc_t *spd, int cpuid);
starpurm_drs_ret_t starpurm_lend_cpus(starpurm_drs_desc_t *spd, int ncpus);
starpurm_drs_ret_t starpurm_lend_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

starpurm_drs_ret_t starpurm_reclaim(starpurm_drs_desc_t *spd);
starpurm_drs_ret_t starpurm_reclaim_cpu(starpurm_drs_desc_t *spd, int cpuid);
starpurm_drs_ret_t starpurm_reclaim_cpus(starpurm_drs_desc_t *spd, int ncpus);
starpurm_drs_ret_t starpurm_reclaim_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

starpurm_drs_ret_t starpurm_acquire(starpurm_drs_desc_t *spd);
starpurm_drs_ret_t starpurm_acquire_cpu(starpurm_drs_desc_t *spd, int cpuid);
starpurm_drs_ret_t starpurm_acquire_cpus(starpurm_drs_desc_t *spd, int ncpus);
starpurm_drs_ret_t starpurm_acquire_cpu_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);

starpurm_drs_ret_t starpurm_return_all(starpurm_drs_desc_t *spd);
starpurm_drs_ret_t starpurm_return_cpu(starpurm_drs_desc_t *spd, int cpuid);

#if 0
/* Pause/resume (not implemented) */
starpurm_drs_ret_t starpurm_create_block_condition(starpurm_block_cond_t *cond);
void starpurm_block_current_task(starpurm_block_cond_t *cond);
void starpurm_signal_block_condition(starpurm_block_cond_t *cond);

void starpurm_register_polling_service(const char *service_name, starpurm_polling_t function, void *data);
void starpurm_unregister_polling_service(const char *service_name, starpurm_polling_t function, void *data);
#endif

/* Devices */
int starpurm_get_device_type_id(const char *type_str);
const char *starpurm_get_device_type_name(int type_id);
int starpurm_get_nb_devices_by_type(int type_id);
int starpurm_get_device_id(int type_id, int device_rank);

starpurm_drs_ret_t starpurm_assign_device_to_starpu(starpurm_drs_desc_t *spd, int type_id, int unit_rank);
starpurm_drs_ret_t starpurm_assign_devices_to_starpu(starpurm_drs_desc_t *spd, int type_id, int ndevices);
starpurm_drs_ret_t starpurm_assign_device_mask_to_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);
starpurm_drs_ret_t starpurm_assign_all_devices_to_starpu(starpurm_drs_desc_t *spd, int type_id);

starpurm_drs_ret_t starpurm_withdraw_device_from_starpu(starpurm_drs_desc_t *spd, int type_id, int unit_rank);
starpurm_drs_ret_t starpurm_withdraw_devices_from_starpu(starpurm_drs_desc_t *spd, int type_id, int ndevices);
starpurm_drs_ret_t starpurm_withdraw_device_mask_from_starpu(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);
starpurm_drs_ret_t starpurm_withdraw_all_devices_from_starpu(starpurm_drs_desc_t *spd, int type_id);

/* --- */

starpurm_drs_ret_t starpurm_lend_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);
starpurm_drs_ret_t starpurm_lend_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices);
starpurm_drs_ret_t starpurm_lend_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);
starpurm_drs_ret_t starpurm_lend_all_devices(starpurm_drs_desc_t *spd, int type_id);

starpurm_drs_ret_t starpurm_reclaim_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);
starpurm_drs_ret_t starpurm_reclaim_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices);
starpurm_drs_ret_t starpurm_reclaim_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);
starpurm_drs_ret_t starpurm_reclaim_all_devices(starpurm_drs_desc_t *spd, int type_id);

starpurm_drs_ret_t starpurm_acquire_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);
starpurm_drs_ret_t starpurm_acquire_devices(starpurm_drs_desc_t *spd, int type_id, int ndevices);
starpurm_drs_ret_t starpurm_acquire_device_mask(starpurm_drs_desc_t *spd, const hwloc_cpuset_t mask);
starpurm_drs_ret_t starpurm_acquire_all_devices(starpurm_drs_desc_t *spd, int type_id);

starpurm_drs_ret_t starpurm_return_all_devices(starpurm_drs_desc_t *spd, int type_id);
starpurm_drs_ret_t starpurm_return_device(starpurm_drs_desc_t *spd, int type_id, int unit_rank);

/* cpusets */
hwloc_cpuset_t starpurm_get_device_worker_cpuset(int type_id, int unit_rank);
hwloc_cpuset_t starpurm_get_global_cpuset(void);
hwloc_cpuset_t starpurm_get_selected_cpuset(void);
hwloc_cpuset_t starpurm_get_all_cpu_workers_cpuset(void);
hwloc_cpuset_t starpurm_get_all_device_workers_cpuset(void);
hwloc_cpuset_t starpurm_get_all_device_workers_cpuset_by_type(int typeid);

#ifdef __cplusplus
}
#endif
#endif /* __STARPURM_H */
