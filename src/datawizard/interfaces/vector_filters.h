#ifndef __VECTOR_FILTERS_H__
#define __VECTOR_FILTERS_H__

#include <datawizard/hierarchy.h>

struct data_state_t;

unsigned block_filter_func_vector(filter *f, struct data_state_t *root_data);
unsigned list_filter_func_vector(filter *f, data_state *root_data);
unsigned divide_in_2_filter_func_vector(filter *f, data_state *root_data);

#endif // __VECTOR_FILTERS_H__
