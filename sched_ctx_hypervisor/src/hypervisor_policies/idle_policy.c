#include "policy_utils.h"

void idle_handle_idle_cycle(unsigned sched_ctx, int worker)
{
	struct sched_ctx_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	struct policy_config *config = sc_w->config;
	if(config != NULL &&  sc_w->current_idle_time[worker] > config->max_idle[worker])
		_resize_to_unknown_receiver(sched_ctx);
}

struct hypervisor_policy idle_policy = {
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = idle_handle_idle_cycle,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.custom = 0,
	.name = "idle"
};
