/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2024-2025 University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_config.h>

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
#define enable_cuda 1
#else
#define enable_cuda 0
#endif
#define MATRIX_REGISTER				\
	unsigned node;							\
	if (starpu_memory_nodes_get_count_by_kind(STARPU_CUDA_RAM) >= 1 && enable_cuda) \
		starpu_memory_node_get_ids_by_type(STARPU_CUDA_RAM, &node, 1); \
	else if (starpu_memory_nodes_get_count_by_kind(STARPU_HIP_RAM) >= 1) \
		starpu_memory_node_get_ids_by_type(STARPU_HIP_RAM, &node, 1); \
	else								\
	{								\
		unsigned nodes[2];					\
		unsigned nram = starpu_memory_node_get_ids_by_type(STARPU_CPU_RAM, nodes, 2); \
		if (nram == 1)						\
			node = nodes[0];				\
		else							\
			node = nodes[1];				\
	}								\
	uintptr_t matrix = starpu_malloc_on_node(node, NX*NY*sizeof(int)); \
	starpu_matrix_data_register(&handle, node, matrix, NX, NX, NY, sizeof(int));

#define MATRIX_FREE						\
	starpu_free_on_node(node, matrix, NX*NY*sizeof(int));
