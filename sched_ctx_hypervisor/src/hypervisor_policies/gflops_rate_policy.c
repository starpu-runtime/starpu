#include "policy_utils.h"

static double _get_total_elapsed_flops_per_sched_ctx(unsigned sched_ctx)
{
	struct sched_ctx_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	double ret_val = 0.0;
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		ret_val += sc_w->total_elapsed_flops[i];
	return ret_val;
}

static double _get_elapsed_flops_per_cpus(struct sched_ctx_wrapper* sc_w, int *ncpus)
{
	double ret_val = 0.0;
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sc_w->sched_ctx);
        int worker;

	if(workers->init_cursor)
                workers->init_cursor(workers);

        while(workers->has_next(workers))
	{
                worker = workers->get_next(workers);
                enum starpu_archtype arch = starpu_worker_get_type(worker);
                if(arch == STARPU_CPU_WORKER)
                {
			ret_val += sc_w->elapsed_flops[worker];
			(*ncpus)++;
                }
        }

	if(workers->init_cursor)
		workers->deinit_cursor(workers);

	return ret_val;
}


double _get_exp_end(unsigned sched_ctx)
{
	struct sched_ctx_wrapper *sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	double elapsed_flops = sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);

	if( elapsed_flops != 0.0)
	{
		double curr_time = starpu_timing_now();
		double elapsed_time = curr_time - sc_w->start_time;
		double exp_end = (elapsed_time * sc_w->remaining_flops /  elapsed_flops) + curr_time;
		return exp_end;
	}
	return -1.0;
}

double _get_ctx_velocity(struct sched_ctx_wrapper* sc_w)
{
        double elapsed_flops = sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);

        if( elapsed_flops != 0.0)
        {
                double curr_time = starpu_timing_now();
                double elapsed_time = curr_time - sc_w->start_time;
                return elapsed_flops/elapsed_time;
        }
}

double _get_cpu_velocity(struct sched_ctx_wrapper* sc_w)
{
        int ncpus = 0;
        double elapsed_flops = _get_elapsed_flops_per_cpus(sc_w, &ncpus);

        if( elapsed_flops != 0.0)
        {
                double curr_time = starpu_timing_now();
                double elapsed_time = curr_time - sc_w->start_time;
                return (elapsed_flops/elapsed_time) / ncpus;
        }

        return -1.0;
}

double _get_flops_left_pct(unsigned sched_ctx)
{
	struct sched_ctx_wrapper *wrapper = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	double total_elapsed_flops = _get_total_elapsed_flops_per_sched_ctx(sched_ctx);
	if(wrapper->total_flops == total_elapsed_flops || total_elapsed_flops > wrapper->total_flops)
		return 0.0;
       
	return (wrapper->total_flops - total_elapsed_flops)/wrapper->total_flops;
}

static int* _get_workers_to_move(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int *nworkers)
{
	struct sched_ctx_wrapper* sender_sc_w = sched_ctx_hypervisor_get_wrapper(sender_sched_ctx);
	struct sched_ctx_wrapper* receiver_sc_w = sched_ctx_hypervisor_get_wrapper(receiver_sched_ctx);
        int *workers = NULL;
        double v_receiver = _get_ctx_velocity(receiver_sc_w);
        double receiver_remainig_flops = receiver_sc_w->remaining_flops;
        double sender_exp_end = _get_exp_end(sender_sched_ctx);
        double sender_v_cpu = _get_cpu_velocity(sender_sc_w);
        double v_for_rctx = (receiver_remainig_flops/(sender_exp_end - starpu_timing_now())) - v_receiver;

        int nworkers_needed = v_for_rctx/sender_v_cpu;
/*      printf("%d->%d: v_rec %lf v %lf v_cpu %lf w_needed %d \n", sender_sched_ctx, receiver_sched_ctx, */
/*             v_receiver, v_for_rctx, sender_v_cpu, nworkers_needed); */
        if(nworkers_needed > 0)
        {
                struct policy_config *sender_config = sched_ctx_hypervisor_get_config(sender_sched_ctx);
                unsigned potential_moving_cpus = _get_potential_nworkers(sender_config, sender_sched_ctx, STARPU_CPU_WORKER);
                unsigned potential_moving_gpus = _get_potential_nworkers(sender_config, sender_sched_ctx, STARPU_CUDA_WORKER);
                unsigned sender_nworkers = starpu_get_nworkers_of_sched_ctx(sender_sched_ctx);
                struct policy_config *config = sched_ctx_hypervisor_get_config(receiver_sched_ctx);
                unsigned nworkers_ctx = starpu_get_nworkers_of_sched_ctx(receiver_sched_ctx);

                if(nworkers_needed < (potential_moving_cpus + 5 * potential_moving_gpus))
                {
                        if((sender_nworkers - nworkers_needed) >= sender_config->min_nworkers)
                        {
                                if((nworkers_ctx + nworkers_needed) > config->max_nworkers)
                                        nworkers_needed = nworkers_ctx > config->max_nworkers ? 0 : (config->max_nworkers - nworkers_ctx);

                                if(nworkers_needed > 0)
                                {
                                        int ngpus = nworkers_needed / 5;
                                        int *gpus;
                                        gpus = _get_first_workers(sender_sched_ctx, &ngpus, STARPU_CUDA_WORKER);
                                        int ncpus = nworkers_needed - ngpus;
                                        int *cpus;
                                        cpus = _get_first_workers(sender_sched_ctx, &ncpus, STARPU_CPU_WORKER);
                                        workers = (int*)malloc(nworkers_needed*sizeof(int));
                                        int i;
					printf("%d: gpus: ", nworkers_needed);
                                        for(i = 0; i < ngpus; i++)
					{
                                                workers[(*nworkers)++] = gpus[i];
						printf("%d ", gpus[i]);
					}
					printf(" cpus:");
                                        for(i = 0; i < ncpus; i++)
					{
                                                workers[(*nworkers)++] = cpus[i];
						printf("%d ", cpus[i]);
					}
					printf("\n");
                                        free(gpus);
                                        free(cpus);
                                }
                        }
                }
		else
                {
//			printf("nworkers_needed = %d\n", nworkers_needed);
                        int nworkers_to_move = _get_nworkers_to_move(sender_sched_ctx);

                        if(sender_nworkers - nworkers_to_move >= sender_config->min_nworkers)
                        {
                                unsigned nshared_workers = starpu_get_nshared_workers(sender_sched_ctx, receiver_sched_ctx);
                                if((nworkers_ctx + nworkers_to_move - nshared_workers) > config->max_nworkers)
                                        nworkers_to_move = nworkers_ctx > config->max_nworkers ? 0 : (config->max_nworkers - nworkers_ctx + nshared_workers);

                                if(nworkers_to_move > 0)
                                {
                                        workers = _get_first_workers(sender_sched_ctx, &nworkers_to_move, -1);
                                        *nworkers = nworkers_to_move;
                                }
                        }
                }
        }
        return workers;
}

static unsigned _gflops_rate_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize)
{
        int ret = 1;
        if(force_resize)
                pthread_mutex_lock(&act_hypervisor_mutex);
        else
                ret = pthread_mutex_trylock(&act_hypervisor_mutex);
        if(ret != EBUSY)
        {
                int nworkers_to_move = 0;
                int *workers_to_move =  _get_workers_to_move(sender_sched_ctx, receiver_sched_ctx, &nworkers_to_move);
		if(nworkers_to_move > 0)
                {
                        sched_ctx_hypervisor_move_workers(sender_sched_ctx, receiver_sched_ctx, workers_to_move, nworkers_to_move);

                        struct policy_config *new_config = sched_ctx_hypervisor_get_config(receiver_sched_ctx);
                        int i;
                        for(i = 0; i < nworkers_to_move; i++)
                                new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;

                        free(workers_to_move);
                }
                pthread_mutex_unlock(&act_hypervisor_mutex);
                return 1;
        }
        return 0;

}

static int _find_fastest_sched_ctx()
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

	double first_exp_end = _get_exp_end(sched_ctxs[0]);
	int fastest_sched_ctx = first_exp_end == -1.0  ? -1 : sched_ctxs[0];
	double curr_exp_end = 0.0;
	int i;
	for(i = 1; i < nsched_ctxs; i++)
	{
		curr_exp_end = _get_exp_end(sched_ctxs[i]);
		if((curr_exp_end < first_exp_end || first_exp_end == -1.0) && curr_exp_end != -1.0)
		{
			first_exp_end = curr_exp_end;
			fastest_sched_ctx = sched_ctxs[i];
		}
	}

	return fastest_sched_ctx;

}

static int _find_slowest_sched_ctx()
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

	int slowest_sched_ctx = -1;
	double curr_exp_end = 0.0;
	double last_exp_end = -1.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		curr_exp_end = _get_exp_end(sched_ctxs[i]);
		/*if it hasn't started bc of no ressources give it priority */
		if(curr_exp_end == -1.0)
			return sched_ctxs[i];
		if( curr_exp_end > last_exp_end)
		{
			slowest_sched_ctx = sched_ctxs[i];
			last_exp_end = curr_exp_end;
		}
	}

	return slowest_sched_ctx;

}

static int _find_slowest_available_sched_ctx(unsigned sched_ctx)
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

	int slowest_sched_ctx = -1;
	double curr_exp_end = 0.0;
	double last_exp_end = -1.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		if(sched_ctxs[i] != sched_ctx)
		{
			curr_exp_end = _get_exp_end(sched_ctxs[i]);
			/*if it hasn't started bc of no ressources give it priority */
			if(curr_exp_end == -1.0)
				return sched_ctxs[i];
			if(last_exp_end < curr_exp_end)
			{
				slowest_sched_ctx = sched_ctxs[i];
				last_exp_end = curr_exp_end;
			}
		}
	}

	return slowest_sched_ctx;

}

static void gflops_rate_resize(unsigned sched_ctx)
{
	double exp_end = _get_exp_end(sched_ctx);
	double flops_left_pct = _get_flops_left_pct(sched_ctx);

	if(flops_left_pct == 0.0f)
	{
		int slowest_sched_ctx = _find_slowest_available_sched_ctx(sched_ctx);
		if(slowest_sched_ctx != -1)
		{
			double slowest_flops_left_pct = _get_flops_left_pct(slowest_sched_ctx);
			if(slowest_flops_left_pct != 0.0f)
			{
				struct policy_config* config = sched_ctx_hypervisor_get_config(sched_ctx);
				config->min_nworkers = 0;
				config->max_nworkers = 0;
				printf("ctx %d finished & gives away the res to %d; slow_left %lf\n", sched_ctx, slowest_sched_ctx, slowest_flops_left_pct);
				_resize(sched_ctx, slowest_sched_ctx, 1);
				sched_ctx_hypervisor_stop_resize(slowest_sched_ctx);
			}
		}
	}

	int fastest_sched_ctx = _find_fastest_sched_ctx();
	int slowest_sched_ctx = _find_slowest_sched_ctx();

//	printf("%d %d \n", fastest_sched_ctx, slowest_sched_ctx);
	if(fastest_sched_ctx != -1 && slowest_sched_ctx != -1 && fastest_sched_ctx != slowest_sched_ctx)
	{
		double fastest_exp_end = _get_exp_end(fastest_sched_ctx);
		double slowest_exp_end = _get_exp_end(slowest_sched_ctx);

		if((slowest_exp_end == -1.0 && fastest_exp_end != -1.0) || ((fastest_exp_end + (fastest_exp_end*0.5)) < slowest_exp_end ))
		{
			double fast_flops_left_pct = _get_flops_left_pct(fastest_sched_ctx);
			if(fast_flops_left_pct < 0.8)
				_gflops_rate_resize(fastest_sched_ctx, slowest_sched_ctx, 0);
		}
	}
}

void gflops_rate_handle_poped_task(unsigned sched_ctx, int worker)
{
	gflops_rate_resize(sched_ctx);
}

struct hypervisor_policy gflops_rate_policy = {
	.handle_poped_task = gflops_rate_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL
};
