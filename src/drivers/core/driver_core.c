#include "driver_core.h"
#include <core/policies/sched_policy.h>

int execute_job_on_core(job_t j, struct worker_s *core_args)
{
	int ret;
	tick_t codelet_start, codelet_end;
	tick_t codelet_start_comm, codelet_end_comm;

	unsigned calibrate_model = 0;

	STARPU_ASSERT(j->cl);
	STARPU_ASSERT(j->cl->core_func);

	if (j->cl->model && j->cl->model->benchmarking)
		calibrate_model = 1;

	if (calibrate_model)
		GET_TICK(codelet_start_comm);

	ret = fetch_codelet_input(j->buffers, j->interface,
			j->nbuffers, 0);

	if (calibrate_model)
		GET_TICK(codelet_end_comm);

	if (ret != 0) {
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return TRYAGAIN;
	}

	TRACE_START_CODELET_BODY(j);

	if (calibrate_model)
		GET_TICK(codelet_start);

	cl_func func = j->cl->core_func;
	func(j->interface, j->cl->cl_arg);
	
	if (calibrate_model)
		GET_TICK(codelet_end);

	TRACE_END_CODELET_BODY(j);

	push_codelet_output(j->buffers, j->nbuffers, 0);

//#ifdef MODEL_DEBUG
	if (calibrate_model)
	{
		double measured = timing_delay(&codelet_start, &codelet_end);
		double measured_comm = timing_delay(&codelet_start_comm, &codelet_end_comm);

//		fprintf(stderr, "%d\t%d\n", (int)j->penality, (int)measured_comm);
		core_args->jobq->total_computation_time += measured;
		core_args->jobq->total_communication_time += measured_comm;

		update_perfmodel_history(j, core_args->arch, measured);
	}
//#endif

	return OK;
}

void *core_worker(void *arg)
{
	struct worker_s *core_arg = arg;

        int core = core_arg->id;


#ifdef USE_FXT
	fxt_register_thread(core_arg->bindid);
#endif
	TRACE_NEW_WORKER(FUT_CORE_KEY);

#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask; 
	CPU_ZERO(&aff_mask);
	CPU_SET(core_arg->bindid, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

        fprintf(stderr, "core worker %d is ready on logical core %d\n", core, core_arg->bindid);

	set_local_memory_node_key(&core_arg->memory_node);

	set_local_queue(core_arg->jobq);

	/* this is only useful (and meaningful) is there is a single
	   memory node "related" to that queue */
	core_arg->jobq->memory_node = core_arg->memory_node;

	core_arg->jobq->total_computation_time = 0.0;
	core_arg->jobq->total_communication_time = 0.0;
	
        /* tell the main thread that we are ready */
        sem_post(&core_arg->ready_sem);

        job_t j;
	int res;

	while (machine_is_running())
	{
                j = pop_task();
                if (j == NULL) continue;

		/* can a core perform that task ? */
		if (!CORE_MAY_PERFORM(j)) 
		{
			/* put it and the end of the queue ... XXX */
			push_task(j);
			continue;
		}

                res = execute_job_on_core(j, core_arg);
		if (res != OK) {
			switch (res) {
				case FATAL:
					assert(0);
				case TRYAGAIN:
					push_task(j);
					continue;
				default: 
					assert(0);
			}
		}

		handle_job_termination(j);

		job_delete(j);
        }

#ifdef DATA_STATS
	if (get_env_number("CALIBRATE") != -1)
		fprintf(stderr, "CORE #%d computation %le comm %le (%lf \%%)\n", core_arg->id, core_arg->jobq->total_computation_time, core_arg->jobq->total_communication_time,  core_arg->jobq->total_communication_time*100.0/core_arg->jobq->total_computation_time);

#endif

	TRACE_WORKER_TERMINATED(FUT_CORE_KEY);

	pthread_exit(NULL);
}
