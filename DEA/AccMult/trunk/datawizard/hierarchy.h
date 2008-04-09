#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include "datawizard/coherency.h"

typedef struct filter_t {
	unsigned (*filter_func)(struct filter_t *, data_state *); /* the actual partitionning function */
	uint32_t filter_arg;
} filter;

void monitor_new_data(data_state *state, uint32_t home_node,
     uintptr_t ptr, uint32_t ld, uint32_t nx, uint32_t ny, size_t elemsize);

void partition_data(data_state *initial_data, filter *f); 
void unpartition_data(data_state *root_data, uint32_t gathering_node);

void map_filter(data_state *root_data, filter *f);

#endif
