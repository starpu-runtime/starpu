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
#include <common/starpu-spinlock.h>

#define DATA_REQ_ALLOCATE	(1<<0)
#define DATA_REQ_COPY		(1<<1)

struct starpu_data_state_t;

LIST_TYPE(data_request,
	starpu_spinlock_t lock;
	unsigned refcnt;

	/* parameters to define the type of request */
	unsigned flags;

	struct starpu_data_state_t *state;
	uint32_t src_node;
	uint32_t dst_node;

	uint32_t handling_node;

	uint8_t read;
	uint8_t write;

	starpu_async_channel async_channel;

	unsigned completed;
	int retval;

	/* in case we have a chain of request (eg. for nvidia multi-GPU) */
	struct data_request_s *next_req[MAXNODES];
	/* who should perform the next request ? */
	unsigned next_req_count;

	/* is StarPU forced to honour that request ? (not really when
	 * prefetching for instance) */
	unsigned strictness;
	unsigned is_a_prefetch_request;

#ifdef USE_FXT
	unsigned com_id;
#endif
);

void init_data_request_lists(void);
void deinit_data_request_lists(void);
void post_data_request(data_request_t r, uint32_t handling_node);
void handle_node_data_requests(uint32_t src_node, unsigned may_alloc);

void handle_pending_node_data_requests(uint32_t src_node);
void handle_all_pending_node_data_requests(uint32_t src_node);

int check_that_no_data_request_exists(uint32_t node);

data_request_t create_data_request(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node, uint32_t handling_node, uint8_t read, uint8_t write, unsigned is_prefetch);
data_request_t search_existing_data_request(struct starpu_data_state_t *state, uint32_t dst_node, uint8_t read, uint8_t write);
int wait_data_request_completion(data_request_t r, unsigned may_alloc);

#endif // __DATA_REQUEST_H__
