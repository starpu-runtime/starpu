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

#ifndef __DATA_REQUEST_H__
#define __DATA_REQUEST_H__

#include <semaphore.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <common/list.h>

struct starpu_data_state_t;

LIST_TYPE(data_request,
	struct starpu_data_state_t *state;
	uint32_t src_node;
	uint32_t dst_node;
	sem_t sem;
	int retval;
);

void init_data_request_lists(void);
int post_data_request(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
void handle_node_data_requests(uint32_t src_node);

#endif // __DATA_REQUEST_H__
