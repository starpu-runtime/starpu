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

#ifndef __STARPU_DATA_H__
#define __STARPU_DATA_H__

#include <starpu_config.h>
#include <starpu-data-interfaces.h>
#include <starpu-data-filters.h>

#define NMAXBUFS        8

struct starpu_data_state_t;

typedef enum {
	R,
	W,
	RW
} starpu_access_mode;

typedef struct starpu_buffer_descr_t {
	starpu_data_handle state;
	starpu_access_mode mode;
} starpu_buffer_descr;

void starpu_unpartition_data(struct starpu_data_state_t *root_data, uint32_t gathering_node);
void starpu_delete_data(struct starpu_data_state_t *state);

void starpu_advise_if_data_is_important(struct starpu_data_state_t *state, unsigned is_important);

void starpu_sync_data_with_mem(struct starpu_data_state_t *state);
void starpu_notify_data_modification(struct starpu_data_state_t *state, uint32_t modifying_node);

void starpu_malloc_pinned_if_possible(float **A, size_t dim);

#endif // __STARPU_DATA_H__
