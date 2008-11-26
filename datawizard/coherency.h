#ifndef __COHERENCY__H__
#define __COHERENCY__H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <common/util.h>
#include <common/mutex.h>
#include <common/rwlock.h>
#include <common/timing.h>
#include <common/fxt.h>

#include "data_parameters.h"
#include "data_request.h"
#include "interfaces/data_interface.h"

typedef enum {
//	MODIFIED,
	OWNER,
	SHARED,
	INVALID
} cache_state;

typedef enum {
	R,
	W,
	RW
} access_mode;

/* this should contain the information relative to a given node */
typedef struct local_data_state_t {
	/* describes the state of the local data in term of coherency */
	cache_state	state; 

	uint32_t refcnt;

	/* is the data locally allocated ? */
	uint8_t allocated; 
	/* was it automatically allocated ? */
	/* perhaps the allocation was perform higher in the hiearchy 
	 * for now this is just translated into !automatically_allocated
	 * */
	uint8_t automatically_allocated;
} local_data_state;

typedef struct data_state_t {
	/* protect the data itself */
	rw_lock	data_lock;
	/* protect meta data */
	mutex header_lock;

	uint32_t nnodes; /* the number of memory nodes that may use it */
	struct data_state_t *children;
	int nchildren;

	/* describe the state of the data in term of coherency */
	local_data_state per_node[MAXNODES];

	/* describe the actual data layout */
	data_interface_t interface[MAXNODES];
	unsigned interfaceid;

	struct data_interface_ops_t *ops;

	/* where is the data home ? -1 if none yet */
	int data_home;

	/* allows special optimization */
	uint8_t is_readonly;
} data_state;

typedef struct buffer_descr_t {
	/* the part used by the runtime */
	data_state *state;
	access_mode mode;

	/* the part given to the kernel */
//	data_interface_t interface;
	unsigned interfaceid;
} buffer_descr;

void display_msi_stats(void);

void display_state(data_state *state);
__attribute__((warn_unused_result))
int fetch_data(data_state *state, access_mode mode);
void release_data(data_state *state, uint32_t write_through_mask);

__attribute__((warn_unused_result))
int _fetch_data(data_state *state, uint32_t requesting_node, uint8_t read, uint8_t write);

uint32_t get_data_refcnt(data_state *state, uint32_t node);

void push_codelet_output(buffer_descr *descrs, unsigned nbuffers, uint32_t mask);

__attribute__((warn_unused_result))
int fetch_codelet_input(buffer_descr *descrs, data_interface_t *interface, unsigned nbuffers, uint32_t mask);

#endif // __COHERENCY__H__
