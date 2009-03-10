#ifndef __COHERENCY__H__
#define __COHERENCY__H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>

#include <common/util.h>
#include <starpu-mutex.h>
#include <common/rwlock.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <common/list.h>

#include <datawizard/data_parameters.h>
#include <datawizard/data_request.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/progress.h>
#include <datawizard/datastats.h>

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

	/* To help the scheduling policies to make some decision, we
	   may keep a track of the tasks that are likely to request 
	   this data on the current node.
	   It is the responsability of the scheduling _policy_ to set that
	   flag when it assigns a task to a queue, policies which do not
	   use this hint can simply ignore it.
	 */
	uint8_t requested;
} local_data_state;

#ifdef NO_DATA_RW_LOCK
/* Everyone that wants to access some piece of data will post a request.
 * Not only StarPU internals, but also the application may put such requests */

LIST_TYPE(data_requester,
	/* what kind of access is requested ? */
	access_mode mode;

	unsigned is_requested_by_codelet;

	/* in case this is a codelet that will do the access */
	struct job_s *j;
	unsigned buffer_index;

	/* if this is more complicated ... (eg. application request) 
	 * NB: this callback is not called with the lock taken !
	 */
	void (*ready_data_callback)(void *argcb);
	void *argcb;
);

#endif

typedef struct data_state_t {
#ifdef NO_DATA_RW_LOCK
	data_requester_list_t req_list;
	/* the number of requests currently in the scheduling engine
	 * (not in the req_list anymore) */
	unsigned refcnt;
	access_mode current_mode;
#else
	/* protect the data itself */
	rw_lock	data_lock;
#endif
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

	/* what is the default write-back mask for that data ? */
	uint32_t wb_mask;

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
//void release_data(data_state *state, uint32_t write_through_mask);

__attribute__((warn_unused_result))
int _fetch_data(data_state *state, uint32_t requesting_node, uint8_t read, uint8_t write);

uint32_t get_data_refcnt(data_state *state, uint32_t node);

void push_codelet_output(buffer_descr *descrs, unsigned nbuffers, uint32_t mask);

__attribute__((warn_unused_result))
int fetch_codelet_input(buffer_descr *descrs, data_interface_t *interface, unsigned nbuffers, uint32_t mask);

void notify_data_modification(data_state *state, uint32_t modifying_node);

int request_data_allocation(data_state *state, uint32_t node);

unsigned is_data_present_or_requested(data_state *state, uint32_t node);

inline void set_data_requested_flag_if_needed(data_state *state, uint32_t node);

void sync_data_with_mem(data_state *state);

#endif // __COHERENCY__H__
