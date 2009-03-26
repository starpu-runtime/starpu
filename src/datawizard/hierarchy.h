#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include <stdarg.h>
#include <datawizard/coherency.h>
#include <datawizard/memalloc.h>

#include <starpu.h>

void monitor_new_data(struct data_state_t *state, uint32_t home_node, uint32_t wb_mask);
void delete_data(struct data_state_t *state);

void advise_if_data_is_important(data_state *state, unsigned is_important);

#endif
