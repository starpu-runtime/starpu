# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
s/\bstruct starpu_per_arch_perfmodel\b/struct starpu_perfmodel_per_arch/g
s/\bstruct starpu_regression_model\b/struct starpu_perfmodel_regression_model/g
s/\bstruct starpu_history_table\b/struct starpu_perfmodel_history_table/g
s/\bstruct starpu_history_entry\b/struct starpu_perfmodel_history_entry/g
s/\bstruct starpu_history_list\b/struct starpu_perfmodel_history_list/g
s/\bstarpu_list_models\b/starpu_perfmodel_list/g
s/\bstruct starpu_model_list\b/struct _starpu_perfmodel_list/g
s/\bstarpu_load_history_debug\b/starpu_perfmodel_load_symbol/g
s/\bstarpu_access_mode\b/enum starpu_access_mode/g
s/\bstruct starpu_codelet_t\b/struct starpu_codelet/g
s/\bstarpu_codelet\b/struct starpu_codelet/g
s/\bstarpu_codelet_t\b/struct starpu_codelet/g
s/\bstarpu_data_handle\b/starpu_data_handle_t/g
s/\bstarpu_block_interface_t\b/struct starpu_block_interface/g
s/\bstarpu_matrix_interface_t\b/struct starpu_matrix_interface/g
s/\bstarpu_vector_interface_t\b/struct starpu_vector_interface/g
s/\bstarpu_variable_interface_t\b/struct starpu_variable_interface/g
s/\bstarpu_csr_interface_t\b/struct starpu_csr_interface/g
s/\bstarpu_bcsr_interface_t\b/struct starpu_bcsr_interface/g
s/\bstarpu_multiformat_interface_t\b/struct starpu_multiformat_interface/g
s/\bstarpu_machine_topology_s\b/starpu_machine_topology/g
s/\bstarpu_htbl32_node_s\b/starpu_htbl32_node/g
s/\bstarpu_history_list_t\b/starpu_history_list/g
s/\bstarpu_buffer_descr_t\b/starpu_buffer_descr/g
s/\bstarpu_history_entry_t\b/starpu_history_entry/g
s/\bstarpu_history_list_t\b/starpu_history_list/g
s/\bstarpu_model_list_t\b/starpu_model_list/g
s/\bstarpu_regression_model_t\b/starpu_regression_model/g
s/\bstarpu_per_arch_perfmodel_t\b/starpu_per_arch_perfmodel/g
#s/\bstarpu_buffer_descr\b/struct starpu_buffer_descr/g
s/\bstarpu_perfmodel_t\b/starpu_perfmodel/g
s/\bstarpu_sched_policy_s\b/starpu_sched_policy/g
s/\bstarpu_data_interface_ops_t\b/starpu_data_interface_ops/g
s/\bstarpu_submit_task\b/starpu_task_submit/g
s/\bstarpu_wait_task\b/starpu_task_wait/g
s/\bstarpu_helper_init_cublas\b/starpu_helper_cublas_init/g
s/\bstarpu_helper_shutdown_cublas\b/starpu_helper_cublas_shutdown/g
s/\bstarpu_deregister_progression_hook\b/starpu_progression_hook_deregister/g
s/\bstarpu_register_progression_hook\b/starpu_progression_hook_register/g
s/\bstarpu_get_worker_id\b/starpu_worker_get_id/g
s/\bstarpu_get_worker_devid\b/starpu_worker_get_devid/g
s/\bstarpu_get_worker_memory_node\b/starpu_worker_get_memory_node/g
s/\bstarpu_get_worker_name\b/starpu_worker_get_name/g
s/\bstarpu_get_worker_type\b/starpu_worker_get_type/g
s/\bstarpu_get_worker_count\b/starpu_worker_get_count/g
s/\bstarpu_get_cpu_worker_count\b/starpu_cpu_worker_get_count/g
s/\bstarpu_get_spu_worker_count\b/starpu_spu_worker_get_count/g
s/\bstarpu_get_opencl_worker_count\b/starpu_opencl_worker_get_count/g
s/\bstarpu_get_cuda_worker_count\b/starpu_cuda_worker_get_count/g
s/\bstarpu_get_local_cuda_stream\b/starpu_cuda_get_local_stream/g
s/\bstarpu_wait_all_tasks\b/starpu_task_wait_for_all/g
s/\bstarpu_delete_data\b/starpu_data_unregister/g
s/\bstarpu_malloc_pinned_if_possible\b/starpu_data_malloc_pinned_if_possible/g
s/\bstarpu_free_pinned_if_possible\b/starpu_data_free_pinned_if_possible/g
s/\bstarpu_sync_data_with_mem\b/starpu_data_acquire/g
s/\bstarpu_data_sync_with_mem\b/starpu_data_acquire/g
s/\bstarpu_sync_data_with_mem_non_blocking\b/starpu_data_acquire_cb/g
s/\bstarpu_data_sync_with_mem_non_blocking\b/starpu_data_acquire_cb/g
s/\bstarpu_release_data_from_mem\b/starpu_data_release/g
s/\bstarpu_data_release_from_mem\b/starpu_data_release/g
s/\bstarpu_advise_if_data_is_important\b/starpu_data_advise_as_important/g
s/\bstarpu_request_data_allocation\b/starpu_data_request_allocation/g
s/\bstarpu_prefetch_data_on_node\b/starpu_data_prefetch_on_node/g
s/\bstarpu_get_sub_data\b/starpu_data_get_sub_data/g
s/\bstarpu_partition_data\b/starpu_data_partition/g
s/\bstarpu_unpartition_data\b/starpu_data_unpartition/g
s/\bstarpu_map_filters\b/starpu_data_map_filters/g
s/\bstarpu_test_if_data_is_allocated_on_node\b/starpu_data_test_if_allocated_on_node/g
s/\bstarpu_get_block_elemsize\b/starpu_block_get_elemsize/g
s/\bstarpu_get_block_local_ldy\b/starpu_block_get_local_ldy/g
s/\bstarpu_get_block_local_ldz\b/starpu_block_get_local_ldz/g
s/\bstarpu_get_block_local_ptr\b/starpu_block_get_local_ptr/g
s/\bstarpu_get_block_nx\b/starpu_block_get_nx/g
s/\bstarpu_get_block_ny\b/starpu_block_get_ny/g
s/\bstarpu_get_block_nz\b/starpu_block_get_nz/g
s/\bstarpu_register_block_data\b/starpu_block_data_register/g
s/\bstarpu_get_bcsr_c\b/starpu_bcsr_get_c/g
s/\bstarpu_get_bcsr_elemsize\b/starpu_bcsr_get_elemsize/g
s/\bstarpu_get_bcsr_firstentry\b/starpu_bcsr_get_firstentry/g
s/\bstarpu_get_bcsr_local_colind\b/starpu_bcsr_get_local_colind/g
s/\bstarpu_get_bcsr_local_nzval\b/starpu_bcsr_get_local_nzval/g
s/\bstarpu_get_bcsr_local_rowptr\b/starpu_bcsr_get_local_rowptr/g
s/\bstarpu_get_bcsr_nnz\b/starpu_bcsr_get_nnz/g
s/\bstarpu_get_bcsr_nrow\b/starpu_bcsr_get_nrow/g
s/\bstarpu_get_bcsr_r\b/starpu_bcsr_get_r/g
s/\bstarpu_register_bcsr_data\b/starpu_bcsr_data_register/g
s/\bstarpu_get_csr_elemsize\b/starpu_csr_get_elemsize/g
s/\bstarpu_get_csr_firstentry\b/starpu_csr_get_firstentry/g
s/\bstarpu_get_csr_local_colind\b/starpu_csr_get_local_colind/g
s/\bstarpu_get_csr_local_nzval\b/starpu_csr_get_local_nzval/g
s/\bstarpu_get_csr_local_rowptr\b/starpu_csr_get_local_rowptr/g
s/\bstarpu_get_csr_nnz\b/starpu_csr_get_nnz/g
s/\bstarpu_get_csr_nrow\b/starpu_csr_get_nrow/g
s/\bstarpu_register_csr_data\b/starpu_csr_data_register/g
s/\bstarpu_get_matrix_elemsize\b/starpu_matrix_get_elemsize/g
s/\bstarpu_get_matrix_local_ld\b/starpu_matrix_get_local_ld/g
s/\bstarpu_get_matrix_local_ptr\b/starpu_matrix_get_local_ptr/g
s/\bstarpu_get_matrix_nx\b/starpu_matrix_get_nx/g
s/\bstarpu_get_matrix_ny\b/starpu_matrix_get_ny/g
s/\bstarpu_register_matrix_data\b/starpu_matrix_data_register/g
s/\bstarpu_divide_in_2_filter_func_vector\b/starpu_vector_divide_in_2_filter_func/g
s/\bstarpu_register_vector_data\b/starpu_vector_data_register/g
s/\bstarpu_get_vector_elemsize\b/starpu_vector_get_elemsize/g
s/\bstarpu_get_vector_local_ptr\b/starpu_vector_get_local_ptr/g
s/\bstarpu_get_vector_nx\b/starpu_vector_get_nx/g
s/\bstarpu_data_set_wb_mask\b/starpu_data_set_wt_mask/g
s/\bstarpu_list_filter_func_vector\b/starpu_vector_list_filter_func/g
s/\bSTARPU_GET_MATRIX_PTR\b/STARPU_MATRIX_GET_PTR/g
s/\bSTARPU_GET_MATRIX_NX\b/STARPU_MATRIX_GET_NX/g
s/\bSTARPU_GET_MATRIX_NY\b/STARPU_MATRIX_GET_NY/g
s/\bSTARPU_GET_MATRIX_LD\b/STARPU_MATRIX_GET_LD/g
s/\bSTARPU_GET_MATRIX_ELEMSIZE\b/STARPU_MATRIX_GET_ELEMSIZE/g
s/\bSTARPU_GET_BLOCK_PTR\b/STARPU_BLOCK_GET_PTR/g
s/\bSTARPU_GET_BLOCK_NX\b/STARPU_BLOCK_GET_NX/g
s/\bSTARPU_GET_BLOCK_NY\b/STARPU_BLOCK_GET_NY/g
s/\bSTARPU_GET_BLOCK_NZ\b/STARPU_BLOCK_GET_NZ/g
s/\bSTARPU_GET_BLOCK_LDY\b/STARPU_BLOCK_GET_LDY/g
s/\bSTARPU_GET_BLOCK_LDZ\b/STARPU_BLOCK_GET_LDZ/g
s/\bSTARPU_GET_BLOCK_ELEMSIZE\b/STARPU_BLOCK_GET_ELEMSIZE/g
s/\bSTARPU_GET_VECTOR_PTR\b/STARPU_VECTOR_GET_PTR/g
s/\bSTARPU_GET_VECTOR_NX\b/STARPU_VECTOR_GET_NX/g
s/\bSTARPU_GET_VECTOR_ELEMSIZE\b/STARPU_VECTOR_GET_ELEMSIZE/g
s/\bSTARPU_GET_VARIABLE_PTR\b/STARPU_VARIABLE_GET_PTR/g
s/\bSTARPU_GET_VARIABLE_ELEMSIZE\b/STARPU_VARIABLE_GET_ELEMSIZE/g
s/\bSTARPU_GET_CSR_NNZ\b/STARPU_CSR_GET_NNZ/g
s/\bSTARPU_GET_CSR_NROW\b/STARPU_CSR_GET_NROW/g
s/\bSTARPU_GET_CSR_NZVAL\b/STARPU_CSR_GET_NZVAL/g
s/\bSTARPU_GET_CSR_COLIND\b/STARPU_CSR_GET_COLIND/g
s/\bSTARPU_GET_CSR_ROWPTR\b/STARPU_CSR_GET_ROWPTR/g
s/\bSTARPU_GET_CSR_FIRSTENTRY\b/STARPU_CSR_GET_FIRSTENTRY/g
s/\bSTARPU_GET_CSR_ELEMSIZE\b/STARPU_CSR_GET_ELEMSIZE/g
s/\bstarpu_print_bus_bandwidth\b/starpu_bus_print_bandwidth/g
s/\bstarpu_get_handle_interface_id\b/starpu_handle_get_interface_id/g
s/\bstarpu_get_current_task\b/starpu_task_get_current/g
s/\bstarpu_pack_cl_args\b/starpu_codelet_pack_args/g
s/\bstarpu_unpack_cl_args\b/starpu_codelet_unpack_args/g
s/\bstarpu_task_deinit\b/starpu_task_clean/g

s/\bstarpu_helper_cublas_init\b/starpu_cublas_init/g
s/\bstarpu_helper_cublas_shutdown\b/starpu_cublas_shutdown/g

s/\bstarpu_allocate_buffer_on_node\b/starpu_malloc_on_node/g
s/\bstarpu_free_buffer_on_node\b/starpu_free_on_node/g

s/\benum starpu_access_mode\b/enum starpu_data_access_mode/g
s/\bstruct starpu_buffer_descr\b/struct starpu_data_descr/g
s/\bstarpu_memory_display_stats\b/starpu_data_display_memory_stats/g
s/\bstarpu_handle_to_pointer\b/starpu_data_handle_to_pointer/g
s/\bstarpu_handle_get_local_ptr\b/starpu_data_get_local_ptr/g
s/\bstarpu_crc32_be_n\b/starpu_hash_crc32c_be_n/g
s/\bstarpu_crc32_be\b/starpu_hash_crc32c_be/g
s/\bstarpu_crc32_string\b/starpu_hash_crc32c_string/g
s/\benum starpu_perf_archtype\b/enum starpu_perfmodel_archtype/g
s/\bstarpu_history_based_expected_perf\b/starpu_permodel_history_based_expected_perf/g
s/\bstruct starpu_task_profiling_info\b/struct starpu_profiling_task_info/g
s/\bstruct starpu_worker_profiling_info\b/struct starpu_profiling_worker_info/g
s/\bstruct starpu_bus_profiling_info\b/struct starpu_profiling_bus_info/g
s/\bstarpu_set_profiling_id\b/starpu_profiling_set_id/g
s/\bstarpu_worker_get_profiling_info\b/starpu_profiling_worker_get_info/g
s/\bstarpu_bus_profiling_helper_display_summary\b/starpu_profiling_bus_helper_display_summary/g
s/\bstarpu_worker_profiling_helper_display_summary\b/starpu_profiling_worker_helper_display_summary/g
s/\benum starpu_archtype\b/enum starpu_worker_archtype/g

s/\bstarpu_handle_get_interface_id\b/starpu_data_get_interface_id/g
s/\bstarpu_handle_get_size\b/starpu_data_get_size/g
s/\bstarpu_handle_pack_data\b/starpu_data_pack/g
s/\bstarpu_handle_unpack_data\b/starpu_data_unpack/g
