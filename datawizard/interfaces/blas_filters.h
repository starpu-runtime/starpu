#ifndef __BLAS_FILTERS_H__
#define __BLAS_FILTERS_H__

#include <datawizard/hierarchy.h>

struct data_state_t;

unsigned block_filter_func(filter *f, struct data_state_t *root_data);
unsigned vertical_block_filter_func(filter *f, struct data_state_t *root_data);

#endif // __BLAS_FILTERS_H__
