#ifndef __MEMALLOC_H__
#define __MEMALLOC_H__

#include <common/list.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/progress.h>

struct data_state_t;

LIST_TYPE(mem_chunk,
	struct data_state_t *data;
	size_t size;

	uint32_t footprint;
	
	/* The footprint of the data is not sufficient to determine whether two
	 * pieces of data have the same layout (there could be collision in the
	 * hash function ...) so we still keep a copy of the actual layout (ie.
	 * the data_interface_t) to stay on the safe side. We make a copy of
	 * because when a data is deleted, the memory chunk remains.
	 */
	struct data_interface_ops_t *ops;
	data_interface_t interface;
	unsigned automatically_allocated;
	unsigned data_was_deleted;
);

void init_mem_chunk_lists(void);
void request_mem_chunk_removal(struct data_state_t *state, unsigned node);
int allocate_memory_on_node(struct data_state_t *state, uint32_t dst_node);
size_t liberate_memory_on_node(mem_chunk_t mc, uint32_t node);

#endif
