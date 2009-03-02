#ifndef __DW_WRITE_BACK_H__
#define __DW_WRITE_BACK_H__

#include <datawizard/coherency.h>

void write_through_data(data_state *state, uint32_t requesting_node, 
					   uint32_t write_through_mask);
void data_set_wb_mask(data_state *state, uint32_t wb_mask);


#endif // __DW_WRITE_BACK_H__
