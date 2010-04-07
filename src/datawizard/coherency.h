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

#ifndef __COHERENCY__H__
#define __COHERENCY__H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>

#include <starpu.h>

#include <pthread.h>
#include <common/starpu_spinlock.h>
#include <common/rwlock.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <common/list.h>
#include <common/config.h>

#include <datawizard/data_request.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/progress.h>
#include <datawizard/datastats.h>

typedef enum {
	STARPU_OWNER,
	STARPU_SHARED,
	STARPU_INVALID
} starpu_cache_state;

/* this should contain the information relative to a given node */
typedef struct starpu_local_data_state_t {
	/* describes the state of the local data in term of coherency */
	starpu_cache_state	state; 

	uint32_t refcnt;

	/* is the data locally allocated ? */
	uint8_t allocated; 
	/* was it automatically allocated ? */
	/* perhaps the allocation was perform higher in the hiearchy 
	 * for now this is just translated into !automatically_allocated
	 * */
	uint8_t automatically_allocated;

	/* To help the scheduling policies to make some decision, we
	   may keep a track of the tasks that are likely to request 
	   this data on the current node.
	   It is the responsability of the scheduling _policy_ to set that
	   flag when it assigns a task to a queue, policies which do not
	   use this hint can simply ignore it.
	 */
	uint8_t requested;
	struct starpu_data_request_s *request;
} starpu_local_data_state;

struct starpu_data_requester_list_s;

struct starpu_data_state_t {
	struct starpu_data_requester_list_s *req_list;
	/* the number of requests currently in the scheduling engine
	 * (not in the req_list anymore) */
	unsigned refcnt;
	starpu_access_mode current_mode;
	/* protect meta data */
	starpu_spinlock_t header_lock;

	uint32_t nnodes; /* the number of memory nodes that may use it */

	/* In case we user filters, the handle may describe a sub-data */
	struct starpu_data_state_t *root_handle; /* root of the tree */
	struct starpu_data_state_t *father_handle; /* father of the node, NULL if the current node is the root */
	unsigned sibling_index; /* indicate which child this node is from the father's perpsective (if any) */
	unsigned depth; /* what's the depth of the tree ? */

	struct starpu_data_state_t *children;
	unsigned nchildren;

	/* describe the state of the data in term of coherency */
	starpu_local_data_state per_node[STARPU_MAXNODES];

	/* describe the actual data layout */
	void *interface[STARPU_MAXNODES];

	struct starpu_data_interface_ops_t *ops;

	/* where is the data home ? -1 if none yet */
	int data_home;

	/* what is the default write-back mask for that data ? */
	uint32_t wb_mask;

	/* allows special optimization */
	uint8_t is_readonly;

	/* in some case, the application may explicitly tell StarPU that a
 	 * piece of data is not likely to be used soon again */
	unsigned is_not_important;
};

void _starpu_display_msi_stats(void);

__attribute__((warn_unused_result))
int _starpu_fetch_data_on_node(struct starpu_data_state_t *state, uint32_t requesting_node, uint8_t read, uint8_t write, unsigned is_prefetch);
void _starpu_release_data_on_node(struct starpu_data_state_t *state, uint32_t default_wb_mask, unsigned memory_node);

void _starpu_update_data_state(struct starpu_data_state_t *state, uint32_t requesting_node, uint8_t write);

uint32_t _starpu_get_data_refcnt(struct starpu_data_state_t *state, uint32_t node);

void _starpu_push_task_output(struct starpu_task *task, uint32_t mask);

__attribute__((warn_unused_result))
int _starpu_fetch_task_input(struct starpu_task *task, uint32_t mask);

unsigned _starpu_is_data_present_or_requested(struct starpu_data_state_t *state, uint32_t node);

inline void _starpu_set_data_requested_flag_if_needed(struct starpu_data_state_t *state, uint32_t node);

int _starpu_prefetch_task_input_on_node(struct starpu_task *task, uint32_t node);

uint32_t _starpu_select_node_to_handle_request(uint32_t src_node, uint32_t dst_node);
uint32_t _starpu_select_src_node(struct starpu_data_state_t *state);

#endif // __COHERENCY__H__
