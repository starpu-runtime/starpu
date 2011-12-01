/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#define _STARPU_DEPRECATED_API_H__

#ifdef __cplusplus
extern "C"
{
#endif

#warning deprecated types. Please update your code to use the latest API, e.g. using tools/dev/rename.sh

#define starpu_codelet			struct starpu_codelet
#define starpu_data_handle		starpu_data_handle_t
#define starpu_block_interface_t	struct starpu_block_interface
#define starpu_matrix_interface_t	struct starpu_matrix_interface
#define starpu_vector_interface_t	struct starpu_vector_interface
#define starpu_variable_interface_t	struct starpu_variable_interface
#define starpu_csr_interface_t		struct starpu_csr_interface
#define starpu_bcsr_interface_t		struct starpu_bcsr_interface
#define starpu_multiformat_interface_t	struct starpu_multiformat_interface
#define starpu_machine_topology_s	starpu_machine_topology
#define starpu_htbl32_node_s		starpu_htbl32_node
#define starpu_history_list_t		starpu_history_list
#define starpu_buffer_descr_t		starpu_buffer_descr
#define starpu_history_entry_t 		starpu_history_entry
#define starpu_history_list_t		starpu_history_list
#define starpu_model_list_t		starpu_model_list
#define starpu_regression_model_t	starpu_regression_model
#define starpu_per_arch_perfmodel_t	starpu_per_arch_perfmodel
#define starpu_buffer_descr		struct starpu_buffer_descr
#define starpu_perfmodel_t		starpu_perfmodel
#define starpu_sched_policy_s		starpu_sched_policy
#define starpu_data_interface_ops_t	starpu_data_interface_ops
#define starpu_access_mode		enum starpu_access_mode

#ifdef __cplusplus
}
#endif

#endif /* _STARPU_DEPRECATED_API_H__ */
