#ifndef __STARPU_DATA_FILTERS_H__
#define __STARPU_DATA_FILTERS_H__

struct data_state_t;

typedef struct filter_t {
	unsigned (*filter_func)(struct filter_t *, struct data_state_t *); /* the actual partitionning function */
	uint32_t filter_arg;
	void *filter_arg_ptr;
} filter;

void starpu_partition_data(struct data_state_t *initial_data, filter *f); 
void starpu_unpartition_data(struct data_state_t *root_data, uint32_t gathering_node);

/* unsigned list */
struct data_state_t *get_sub_data(struct data_state_t *root_data, unsigned depth, ... );

/* filter * list */
void starpu_map_filters(struct data_state_t *root_data, unsigned nfilters, ...);

/* a few examples of filters */

/* for BCSR */
unsigned canonical_block_filter_bcsr(filter *f, struct data_state_t *root_data);
unsigned vertical_block_filter_func_csr(filter *f, struct data_state_t *root_data);
/* (filters for BLAS interface) */
unsigned block_filter_func(filter *f, struct data_state_t *root_data);
unsigned vertical_block_filter_func(filter *f, struct data_state_t *root_data);

/* for vector */
unsigned block_filter_func_vector(filter *f, struct data_state_t *root_data);
unsigned list_filter_func_vector(filter *f, struct data_state_t *root_data);
unsigned divide_in_2_filter_func_vector(filter *f, struct data_state_t *root_data);

#endif
