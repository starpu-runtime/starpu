#ifndef __DATA_REQUEST_H__
#define __DATA_REQUEST_H__

#include <semaphore.h>
#include <datawizard/datawizard.h>
#include <common/list.h>

LIST_TYPE(data_request,
	data_state *state;
	uint32_t src_node;
	uint32_t dst_node;
	sem_t sem;
	int retval;
);

void init_data_request_lists(void);
int post_data_request(data_state *state, uint32_t src_node, uint32_t dst_node);
void handle_node_data_requests(uint32_t src_node);

#endif // __DATA_REQUEST_H__
