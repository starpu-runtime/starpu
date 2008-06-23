#include "mult_core.h"

extern int corecounters[NMAXCORES];
extern unsigned ncores;

void execute_job_on_core(job_t j)
{
        switch (j->type) {
		case CODELET:
			ASSERT(j->cl);
			ASSERT(j->cl->core_func);
			fetch_codelet_input(j->buffers, j->nbuffers);

			TRACE_START_CODELET_BODY(j);
			j->cl->core_func(j->buffers, j->cl->cl_arg);
			TRACE_END_CODELET_BODY(j);

			push_codelet_output(j->buffers, j->nbuffers, 0);
			break;
                case ABORT:
                        fprintf(stderr, "core abort\n");
                        thread_exit(NULL);
                        break;
                default:
			fprintf(stderr, "don't know what to do with that task on a core ! ... \n");
			ASSERT(0);
                        break;
        }
}

void *core_worker(void *arg)
{
        int core = ((core_worker_arg *)arg)->coreid;

#ifdef USE_FXT
	fxt_register_thread(((core_worker_arg *)arg)->bindid);
#endif
	TRACE_NEW_WORKER(FUT_CORE_KEY);

#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask; 
	CPU_ZERO(&aff_mask);
	CPU_SET(((core_worker_arg *)arg)->bindid, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

        fprintf(stderr, "core worker %d is ready on logical core %d\n", core, ((core_worker_arg *)arg)->bindid);

	set_local_memory_node_key(&(((core_worker_arg *)arg)->memory_node));

        /* tell the main thread that we are ready */
        ((core_worker_arg *)arg)->ready_flag = 1;

        job_t j;

        do {
                j = pop_task();
                if (j == NULL) continue;

		/* can a core perform that task ? */
		if (!CORE_MAY_PERFORM(j)) 
		{
			/* put it and the end of the queue ... XXX */
			push_task(j);
			continue;
		}

                execute_job_on_core(j);

                if (j->cb)
                        j->cb(j->argcb);

                corecounters[core]++;

		/* in case there are dependencies, wake up the proper tasks */
		notify_dependencies(&j);

//		job_delete(j);
        } while(1);

        return NULL;
}
