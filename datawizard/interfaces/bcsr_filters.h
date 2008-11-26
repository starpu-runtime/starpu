#ifndef __BCSR_FILTERS_H__
#define __BCSR_FILTERS_H__

#include <datawizard/hierarchy.h>

struct data_state_t;

unsigned canonical_block_filter_bcsr(filter *f, struct data_state_t *root_data);

#endif // __BCSR_FILTERS_H__
