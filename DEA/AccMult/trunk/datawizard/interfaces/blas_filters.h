#ifndef __BLAS_FILTERS_H__
#define __BLAS_FILTERS_H__

#include "../datawizard/hierarchy.h"

unsigned block_filter_func(filter *f, data_state *root_data);
unsigned vertical_block_filter_func(filter *f, data_state *root_data);

#endif // __BLAS_FILTERS_H__
