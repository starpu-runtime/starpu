#ifndef __COPY_DRIVER_H__
#define __COPY_DRIVER_H__

#include "coherency.h"
#include "memalloc.h"

#ifdef USE_CUDA
#include <cublas.h>
#endif


typedef enum {
	UNUSED,
	SPU_LS,
	RAM,
	CUDA_RAM
} node_kind;

typedef struct {
	unsigned nnodes;
	node_kind nodes[MAXNODES];

	/* the list of queues that are attached to a given node */
	// XXX 32 is set randomly !
	struct jobq_s *attached_queues[MAXNODES][32];
	/* the number of queues attached to each node */
	unsigned queues_count[MAXNODES];
} mem_node_descr;

struct data_state_t;

__attribute__((warn_unused_result))
int driver_copy_data(struct data_state_t *state, uint32_t src_node_mask, uint32_t dst_node, unsigned donotread);

void init_memory_nodes(void);
void set_local_memory_node_key(unsigned *node);
unsigned get_local_memory_node(void);
unsigned register_memory_node(node_kind kind);
void memory_node_attach_queue(struct jobq_s *q, unsigned nodeid);
void wake_all_blocked_workers(void);
void wake_all_blocked_workers_on_node(unsigned nodeid);

node_kind get_node_kind(uint32_t node);

__attribute__((warn_unused_result))
int driver_copy_data_1_to_1(struct data_state_t *state, uint32_t node, 
				uint32_t requesting_node, unsigned donotread);

int allocate_per_node_buffer(struct data_state_t *state, uint32_t node);

#ifdef DATA_STATS
void display_comm_ammounts(void);
#endif

#endif // __COPY_DRIVER_H__
