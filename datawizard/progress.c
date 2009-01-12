#include <pthread.h>
#include <core/workers.h>
#include <datawizard/progress.h>
#include <datawizard/data_request.h>

extern pthread_key_t local_workers_key;

#ifdef USE_GORDON
extern void handle_terminated_job_per_worker(struct worker_s *worker);
#endif

void datawizard_progress(uint32_t memory_node)
{
	/* in case some other driver requested data */
	handle_node_data_requests(memory_node);

#ifdef USE_GORDON
	/* XXX quick and dirty !! */
	struct worker_set_s *set;
	set = pthread_getspecific(local_workers_key);
	if (set) {
		/* make the corresponding workers progress */
		unsigned worker;
		for (worker = 0; worker < set->nworkers; worker++)
		{
			handle_terminated_job_per_worker(&set->workers[worker]);
		}
	}
#endif
}
