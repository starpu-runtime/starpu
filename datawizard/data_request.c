#include <datawizard/data_request.h>

static data_request_list_t data_requests[MAXNODES];
static mutex data_requests_mutex[MAXNODES];

void init_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < MAXNODES; i++)
	{
		data_requests[i] = data_request_list_new();
		init_mutex(&data_requests_mutex[i]);
	}
}

int post_data_request(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	int retvalue;

	data_request_t r = data_request_new();

	r->state = state;
	r->src_node = src_node;
	r->dst_node = dst_node;
	sem_init(&r->sem, 0, 0);

	/* insert the request in the proper list */
	take_mutex(&data_requests_mutex[src_node]);
	data_request_list_push_front(data_requests[src_node], r);
	release_mutex(&data_requests_mutex[src_node]);

	/* wake the threads that could perform that operation */
	wake_all_blocked_workers_on_node(src_node);

	/* wait for the request to be performed */
	//sem_wait(&r->sem);
	//while(sem_trywait(&r->sem) == -1)
	//	wake_all_blocked_workers_on_node(src_node);

	while(sem_trywait(&r->sem) == -1)
		datawizard_progress(dst_node);

	retvalue = r->retval;
	
	/* the request is useless now */
	data_request_delete(r);

	return retvalue;	
}

void handle_node_data_requests(uint32_t src_node)
{
	take_mutex(&data_requests_mutex[src_node]);

	/* for all entries of the list */
	data_request_list_t l = data_requests[src_node];
	data_request_t r;

	while (!data_request_list_empty(l))
	{
		r = data_request_list_pop_back(l);		
		release_mutex(&data_requests_mutex[src_node]);

		/* perform the transfer */
		/* the header of the data must be locked by the worker that submitted the request */
		r->retval = driver_copy_data_1_to_1(r->state, r->src_node, r->dst_node, 0);
		
		/* wake the requesting worker up */
		sem_post(&r->sem);

		take_mutex(&data_requests_mutex[src_node]);
	}

	release_mutex(&data_requests_mutex[src_node]);
}
