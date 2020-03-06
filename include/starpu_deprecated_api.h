/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_DEPRECATED_API_H__
#define __STARPU_DEPRECATED_API_H__

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(STARPU_USE_DEPRECATED_API) || defined(STARPU_USE_DEPRECATED_ONE_ZERO_API)
#warning Your application is using deprecated types. You may want to update to use the latest API, by using tools/dev/rename.sh.
#endif /* defined(STARPU_USE_DEPRECATED_API) || defined(STARPU_USE_DEPRECATED_ONE_ZERO_API) */

#define starpu_permodel_history_based_expected_perf	starpu_perfmodel_history_based_expected_perf

#ifdef STARPU_USE_DEPRECATED_ONE_ZERO_API

#define starpu_allocate_buffer_on_node	starpu_malloc_on_node
#define starpu_free_buffer_on_node	starpu_free_on_node
#define starpu_helper_cublas_init	starpu_cublas_init
#define starpu_helper_cublas_shutdown	starpu_cublas_shutdown

#define starpu_canonical_block_filter_bcsr	starpu_bcsr_filter_canonical_block
#define starpu_vertical_block_filter_func_csr	starpu_csr_filter_vertical_block

#define starpu_block_filter_func			starpu_matrix_filter_block
#define starpu_block_shadow_filter_func			starpu_matrix_filter_block_shadow
#define starpu_vertical_block_filter_func		starpu_matrix_filter_vertical_block
#define starpu_vertical_block_shadow_filter_func	starpu_matrix_filter_vertical_block_shadow

#define starpu_block_filter_func_vector		starpu_vector_filter_block
#define starpu_block_shadow_filter_func_vector	starpu_vector_filter_block_shadow
#define starpu_vector_list_filter_func		starpu_vector_filter_list
#define starpu_vector_divide_in_2_filter_func	starpu_vector_filter_divide_in_2

#define starpu_block_filter_func_block			starpu_block_filter_block
#define starpu_block_shadow_filter_func_block		starpu_block_filter_block_shadow
#define starpu_vertical_block_filter_func_block		starpu_block_filter_vertical_block
#define starpu_vertical_block_shadow_filter_func_block	starpu_block_filter_vertical_block_shadow
#define starpu_depth_block_filter_func_block		starpu_block_filter_depth_block
#define starpu_depth_block_shadow_filter_func_block	starpu_block_filter_depth_block_shadow

#define starpu_display_codelet_stats		starpu_codelet_display_stats

#define starpu_access_mode				starpu_data_access_mode
#define starpu_buffer_descr				starpu_data_descr
#define starpu_memory_display_stats			starpu_data_display_memory_stats
#define starpu_handle_to_pointer			starpu_data_handle_to_pointer
#define starpu_handle_get_local_ptr			starpu_data_get_local_ptr
#define starpu_crc32_be_n				starpu_hash_crc32c_be_n
#define starpu_crc32_be					starpu_hash_crc32c_be
#define starpu_crc32_string				starpu_hash_crc32c_string
#define starpu_perf_archtype				starpu_perfmodel_archtype
#define starpu_history_based_expected_perf		starpu_perfmodel_history_based_expected_perf
#define starpu_task_profiling_info			starpu_profiling_task_info
#define starpu_worker_profiling_info			starpu_profiling_worker_info
#define starpu_bus_profiling_info			starpu_profiling_bus_info
#define starpu_set_profiling_id				starpu_profiling_set_id
#define starpu_worker_get_profiling_info		starpu_profiling_worker_get_info
#define starpu_bus_profiling_helper_display_summary	starpu_profiling_bus_helper_display_summary
#define starpu_worker_profiling_helper_display_summary	starpu_profiling_worker_helper_display_summary
#define starpu_archtype					starpu_worker_archtype

#define starpu_handle_get_interface_id		starpu_data_get_interface_id
#define starpu_handle_get_size			starpu_data_get_size
#define starpu_handle_pack_data			starpu_data_pack
#define starpu_handle_unpack_data		starpu_data_unpack

#endif /* STARPU_USE_DEPRECATED_ONE_ZERO_API */

#ifdef STARPU_USE_DEPRECATED_API
typedef starpu_data_handle_t starpu_data_handle;
typedef struct starpu_block_interface starpu_block_interface_t;
typedef struct starpu_matrix_interface starpu_matrix_interface_t;
typedef struct starpu_vector_interface starpu_vector_interface_t;
typedef struct starpu_variable_interface starpu_variable_interface_t;
typedef struct starpu_csr_interface starpu_csr_interface_t;
typedef struct starpu_bcsr_interface starpu_bcsr_interface_t;
typedef struct starpu_multiformat_interface starpu_multiformat_interface_t;
#define starpu_machine_topology_s starpu_machine_topology
#define starpu_htbl32_node_s starpu_htbl32_node
#define starpu_history_list_t starpu_history_list
#define starpu_buffer_descr_t starpu_buffer_descr
#define starpu_regression_model_t starpu_regression_model
#define starpu_per_arch_perfmodel_t starpu_per_arch_perfmodel
#define starpu_perfmodel_t starpu_perfmodel
#define starpu_sched_policy_s starpu_sched_policy
#define starpu_data_interface_ops_t starpu_data_interface_ops

typedef struct starpu_buffer_descr starpu_buffer_descr;
typedef struct starpu_codelet starpu_codelet;
typedef struct starpu_codelet starpu_codelet_t;
typedef enum starpu_access_mode starpu_access_mode;

#define starpu_print_bus_bandwidth     starpu_bus_print_bandwidth
#define starpu_get_handle_interface_id starpu_handle_get_interface_id
#define starpu_get_current_task        starpu_task_get_current
#define starpu_unpack_cl_args          starpu_codelet_unpack_args
#define starpu_pack_cl_args   	       starpu_codelet_pack_args
#define starpu_task_deinit	       starpu_task_clean

#endif /* STARPU_USE_DEPRECATED_API */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DEPRECATED_API_H__ */
