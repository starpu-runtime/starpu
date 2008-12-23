#include "queues.h"

/*
 * There can be various queue designs
 * 	- trivial single list
 * 	- cilk-like 
 * 	- hierarchical (marcel-like)
 */

void setup_queues(void (*init_queue_design)(void),
		  struct jobq_s *(*func_init_queue)(void), 
		  struct machine_config_s *config) 
{
	unsigned worker;

	init_queue_design();

	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct  worker_s *workerarg = &config->workers[worker];
		
		workerarg->jobq = func_init_queue();

		/* warning : in case there are multiple workers on the same
                   queue, we overwrite this value so that it is meaningless
		 */
		workerarg->jobq->arch = workerarg->perf_arch;

		switch (workerarg->arch) {
			case CORE_WORKER:
				workerarg->jobq->who |= CORE;
				workerarg->jobq->alpha = CORE_ALPHA;
				break;
			case CUDA_WORKER:
				workerarg->jobq->who |= CUDA|CUBLAS;
				workerarg->jobq->alpha = CUDA_ALPHA;
				break;
			default:
				ASSERT(0);
		}
		
		memory_node_attach_queue(workerarg->jobq, workerarg->memory_node);
	}
}

/* this may return NULL for an "anonymous thread" */
struct jobq_s *get_local_queue(void)
{
	struct sched_policy_s *policy = get_sched_policy();

	return pthread_getspecific(policy->local_queue_key);
}

/* XXX how to retrieve policy ? that may be given in the machine config ? */
void set_local_queue(struct jobq_s *jobq)
{
	struct sched_policy_s *policy = get_sched_policy();

	pthread_setspecific(policy->local_queue_key, jobq);
}
