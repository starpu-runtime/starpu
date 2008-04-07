#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include "datawizard/coherency.h"

typedef struct filter_t {
	int (*filter_func)(struct filter_t *, data_state *); /* the actual partitionning function */
	uint32_t filter_arg;
} filter;

void partition_data(data_state *initial_data, filter *f); 

void unpartition_data(data_state *root_data, uint32_t gathering_node);

int block_filter_func(filter *f, data_state *root_data);

#endif
