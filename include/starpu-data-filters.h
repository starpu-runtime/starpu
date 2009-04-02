/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_DATA_FILTERS_H__
#define __STARPU_DATA_FILTERS_H__

struct starpu_data_state_t;

typedef struct starpu_filter_t {
	unsigned (*filter_func)(struct starpu_filter_t *, struct starpu_data_state_t *); /* the actual partitionning function */
	uint32_t filter_arg;
	void *filter_arg_ptr;
} starpu_filter;

void starpu_partition_data(struct starpu_data_state_t *initial_data, starpu_filter *f); 
void starpu_unpartition_data(struct starpu_data_state_t *root_data, uint32_t gathering_node);

/* unsigned list */
struct starpu_data_state_t *get_sub_data(struct starpu_data_state_t *root_data, unsigned depth, ... );

/* starpu_filter * list */
void starpu_map_filters(struct starpu_data_state_t *root_data, unsigned nfilters, ...);

/* a few examples of filters */

/* for BCSR */
unsigned starpu_canonical_block_filter_bcsr(starpu_filter *f, struct starpu_data_state_t *root_data);
unsigned starpu_vertical_block_filter_func_csr(starpu_filter *f, struct starpu_data_state_t *root_data);
/* (filters for BLAS interface) */
unsigned starpu_block_filter_func(starpu_filter *f, struct starpu_data_state_t *root_data);
unsigned starpu_vertical_block_filter_func(starpu_filter *f, struct starpu_data_state_t *root_data);

/* for vector */
unsigned starpu_block_filter_func_vector(starpu_filter *f, struct starpu_data_state_t *root_data);
unsigned starpu_list_filter_func_vector(starpu_filter *f, struct starpu_data_state_t *root_data);
unsigned starpu_divide_in_2_filter_func_vector(starpu_filter *f, struct starpu_data_state_t *root_data);

#endif
