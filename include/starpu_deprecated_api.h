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

typedef starpu_data_handle_t starpu_data_handle;
typedef struct starpu_block_interface starpu_block_interface_t;
typedef struct starpu_matrix_interface starpu_matrix_interface_t;
typedef struct starpu_vector_interface starpu_vector_interface_t;
typedef struct starpu_variable_interface starpu_variable_interface_t;
typedef struct starpu_csr_interface starpu_csr_interface_t;
typedef struct starpu_bcsr_interface starpu_bcsr_interface_t;
typedef struct starpu_multiformat_interface starpu_multiformat_interface_t;
typedef starpu_machine_topology starpu_machine_topology_s;
typedef starpu_htbl32_node starpu_htbl32_node_s;
typedef starpu_history_list starpu_history_list_t;
typedef starpu_buffer_descr starpu_buffer_descr_t;
typedef starpu_history_entry starpu_history_entry_t;
typedef starpu_history_list starpu_history_list_t;
typedef starpu_model_list starpu_model_list_t;
typedef starpu_regression_model starpu_regression_model_t;
typedef starpu_per_arch_perfmodel starpu_per_arch_perfmodel_t;
typedef starpu_perfmodel starpu_perfmodel_t;
typedef starpu_sched_policy starpu_sched_policy_s;
typedef starpu_data_interface_ops starpu_data_interface_ops_t;

typedef struct starpu_buffer_descr starpu_buffer_descr;
typedef struct starpu_codelet starpu_codelet;
typedef enum starpu_access_mode starpu_access_mode;

#ifdef __cplusplus
}
#endif

#endif /* _STARPU_DEPRECATED_API_H__ */
