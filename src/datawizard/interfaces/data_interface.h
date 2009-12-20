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

#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

#include <starpu.h>
#include <common/config.h>
#include <datawizard/data_parameters.h>

#ifdef USE_GORDON
/* to get the gordon_strideSize_t data structure from gordon */
#include <gordon.h>
#endif


struct starpu_data_state_t;

struct data_interface_ops_t {
	size_t (*allocate_data_on_node)(struct starpu_data_state_t *state,
					uint32_t node);
	void (*liberate_data_on_node)(starpu_data_interface_t *interface,
					uint32_t node);
	const struct copy_data_methods_s *copy_methods;
	size_t (*dump_data_interface)(starpu_data_interface_t *interface, 
					void *buffer);
	size_t (*get_size)(struct starpu_data_state_t *state);
	uint32_t (*footprint)(struct starpu_data_state_t *state, uint32_t hstate);
	void (*display)(struct starpu_data_state_t *state, FILE *f);
#ifdef USE_GORDON
	int (*convert_to_gordon)(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif
	/* an identifier that is unique to each interface */
	unsigned interfaceid;
};

#endif // __DATA_INTERFACE_H__
