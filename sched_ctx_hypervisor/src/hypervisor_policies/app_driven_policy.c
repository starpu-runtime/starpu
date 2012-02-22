#include "policy_utils.h"

void app_driven_handle_post_exec_hook(unsigned sched_ctx, struct starpu_htbl32_node_s* resize_requests, int task_tag)
{
	void* sched_ctx_pt =  _starpu_htbl_search_32(resize_requests, (uint32_t)task_tag);
	if(sched_ctx_pt && sched_ctx_pt != resize_requests)
	{
		_resize_to_unknown_receiver(sched_ctx);
		_starpu_htbl_insert_32(&resize_requests, (uint32_t)task_tag, NULL);
	}

}

struct hypervisor_policy app_driven_policy = {
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = app_driven_handle_post_exec_hook
};
