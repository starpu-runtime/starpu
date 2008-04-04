#include "mult_gordon.h"
#include "gordon_interface.h"

void *gordon_worker(void *arg)
{
	struct gordon_worker_arg_t* args = (struct gordon_worker_arg_t*)arg;

#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(args->bindid, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

	/* TODO set_local_memory_node per SPU */

	gordon_init(args->nspus);	

	/* tell the core that gordon is ready */
	args->ready_flag = 1;

	int res;
	job_t j;
	do {
		j = pop_task();
		if (j == NULL) continue;

		/* is gordon able to do that ? */
		if (!GORDON_MAY_PERFORM(j))
		{
			push_task(j);
			continue;
		}

		/* XXX that model is not really appropriated for gordon */
		res = execute_job_on_gordon_non_blocking(j);
		if (res != OK) {
			switch (res) {
				case TRYAGAIN:
					push_task(j);
					break;
				case OK:
				case FATAL:
				default:
					assert(0);
			}
		}

		if (j->cb)
			j->cb(j->argcb);

		/* TODO add counters */

		job_delete(j);

	} while(1);

	return NULL;
}
