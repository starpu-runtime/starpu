#include <datawizard/progress.h>
#include <datawizard/data_request.h>

void datawizard_progress(uint32_t memory_node)
{
	/* in case some other driver requested data */
	handle_node_data_requests(memory_node);
}
