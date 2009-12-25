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

#include <starpu.h>
#include <starpu_config.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct starpu_filter_t {
	void (*filter_func)(struct starpu_filter_t *, starpu_data_handle); /* the actual partitionning function */
	uint32_t filter_arg;
	void *filter_arg_ptr;
} starpu_filter;

void starpu_partition_data(starpu_data_handle initial_data, starpu_filter *f); 
void starpu_unpartition_data(starpu_data_handle root_data, uint32_t gathering_node);

struct data_interface_ops_t;
void starpu_data_create_children(starpu_data_handle handle, unsigned nchildren,
		 struct data_interface_ops_t *children_interface_ops);

starpu_data_handle starpu_data_get_child(starpu_data_handle handle, unsigned i);

/* unsigned list */
starpu_data_handle get_sub_data(starpu_data_handle root_data, unsigned depth, ... );

/* starpu_filter * list */
void starpu_map_filters(starpu_data_handle root_data, unsigned nfilters, ...);

/* a few examples of filters */

/* for BCSR */
void starpu_canonical_block_filter_bcsr(starpu_filter *f, starpu_data_handle root_data);
void starpu_vertical_block_filter_func_csr(starpu_filter *f, starpu_data_handle root_data);
/* (filters for BLAS interface) */
void starpu_block_filter_func(starpu_filter *f, starpu_data_handle root_data);
void starpu_vertical_block_filter_func(starpu_filter *f, starpu_data_handle root_data);

/* for vector */
void starpu_block_filter_func_vector(starpu_filter *f, starpu_data_handle root_data);
void starpu_list_filter_func_vector(starpu_filter *f, starpu_data_handle root_data);
void starpu_divide_in_2_filter_func_vector(starpu_filter *f, starpu_data_handle root_data);

#ifdef __cplusplus
}
#endif

#endif
