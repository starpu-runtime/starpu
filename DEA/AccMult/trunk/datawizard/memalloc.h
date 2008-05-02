#ifndef __MEMALLOC_H__
#define __MEMALLOC_H__

#include <common/list.h>
#include "coherency.h"
#include "copy-driver.h"

LIST_TYPE(mem_chunk,
	data_state *data;
	size_t size;
);

void init_mem_chunk_lists(void);
size_t reclaim_memory(uint32_t node);
void request_mem_chunk_removal(data_state *state, unsigned node);
void allocate_memory_on_node(data_state *state, uint32_t dst_node);


#endif
