#ifndef __COPY_DRIVER_H__
#define __COPY_DRIVER_H__

#include "coherency.h"

#ifdef USE_CUBLAS
#include <cublas.h>
#endif

typedef enum {
	UNUSED,
	SPU_LS,
	RAM,
	CUBLAS_RAM,
	CUDA_RAM
} node_kind;

typedef struct {
	unsigned nnodes;
	node_kind nodes[MAXNODES];
} mem_node_descr;

void driver_copy_data(data_state *state, uint32_t src_node_mask, uint32_t dst_node);

void init_memory_nodes(void);
void set_local_memory_node_key(unsigned *node);
unsigned get_local_memory_node(void);
unsigned register_memory_node(node_kind kind);

#endif // __COPY_DRIVER_H__
