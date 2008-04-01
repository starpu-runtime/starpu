#ifndef __COHERENCY__H__
#define __COHERENCY__H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <common/util.h>

#define MAXNODES	6

#define TAKEN	1
#define FREE	0

typedef enum {
//	MODIFIED,
	OWNER,
	SHARED,
	INVALID
} cache_state;

/* this should contain the information relative to a given node */
typedef struct local_data_state_t {
	/* describes the state of the local data in term of coherency */
	cache_state	state; 
	/* where is that data stored on the local node ? */
	uintptr_t ptr;
	/* is the data locally allocated ? */
	uint8_t allocated; 
	/* was it automatically allocated ? */
	uint8_t automatically_allocated;
} local_data_state;

/* this structure is used for locking purpose (this must take into
 * account the fact that it may be accessed using DMA for instance)*/
typedef struct data_lock_t {
	/* we only have a trivial implementation yet ! */
	volatile uint32_t taken;
} data_lock;

typedef struct data_state_t {
	uint32_t nnodes; /* the number of memory nodes that may use it */
	local_data_state per_node[MAXNODES];
	size_t length; /* XXX remove that when possible !! */
	data_lock lock;
} data_state;

void take_lock(data_lock *lock);
void release_lock(data_lock *lock);
void monitor_new_data(data_state *state, uint32_t home_node, 
			uintptr_t ptr, size_t length);
void display_state(data_state *state);
void copy_data_to_node(data_state *state, uint32_t requesting_node);
uintptr_t fetch_data(data_state *state, uint32_t requesting_node,
			uint8_t read, uint8_t write);

void release_data(data_state *state, uint32_t requesting_node, uint32_t write_through_mask);

#endif // __COHERENCY__H__
