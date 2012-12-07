#include <sched_ctx_hypervisor.h>
#include <pthread.h>

static int _compute_priority(unsigned sched_ctx)
{
	struct policy_config *config = sched_ctx_hypervisor_get_config(sched_ctx);

	int total_priority = 0;

	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);
	int worker;

	if(workers->init_cursor)
		workers->init_cursor(workers);

	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		total_priority += config->priority[worker];
	}

	if(workers->init_cursor)
		workers->deinit_cursor(workers);
	return total_priority;
}

static unsigned _find_poor_sched_ctx(unsigned req_sched_ctx, int nworkers_to_move)
{
	int i;
	int highest_priority = -1;
	int current_priority = 0;
	unsigned sched_ctx = STARPU_NMAX_SCHED_CTXS;
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();


	struct policy_config *config = NULL;

	for(i = 0; i < nsched_ctxs; i++)
	{
		if(sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && sched_ctxs[i] != req_sched_ctx)
		{
			unsigned nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctxs[i]);
			config  = sched_ctx_hypervisor_get_config(sched_ctxs[i]);
			if((nworkers + nworkers_to_move) <= config->max_nworkers)
			{
				current_priority = _compute_priority(sched_ctxs[i]);
				if (highest_priority < current_priority)
				{
					highest_priority = current_priority;
					sched_ctx = sched_ctxs[i];
				}
			}
		}
	}
	
	return sched_ctx;
}

int* _get_first_workers(unsigned sched_ctx, unsigned *nworkers, enum starpu_archtype arch)
{
	struct policy_config *config = sched_ctx_hypervisor_get_config(sched_ctx);

	int *curr_workers = (int*)malloc((*nworkers) * sizeof(int));
	int i;
	for(i = 0; i < *nworkers; i++)
		curr_workers[i] = -1;

	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);
	int index;
	int worker;
	int considered = 0;

	if(workers->init_cursor)
		workers->init_cursor(workers);

	for(index = 0; index < *nworkers; index++)
	{
		while(workers->has_next(workers))
		{
			considered = 0;
			worker = workers->get_next(workers);
			enum starpu_archtype curr_arch = starpu_worker_get_type(worker);
			if(arch == 0 || curr_arch == arch)
			{

				if(!config->fixed_workers[worker])
				{
					for(i = 0; i < index; i++)
					{
						if(curr_workers[i] == worker)
						{
							considered = 1;
							break;
						}
					}
					
					if(!considered)
					{
						/* the first iteration*/
						if(curr_workers[index] < 0)
						curr_workers[index] = worker;
						/* small priority worker is the first to leave the ctx*/
						else if(config->priority[worker] <
							config->priority[curr_workers[index]])
						curr_workers[index] = worker;
						/* if we don't consider priorities check for the workers
						   with the biggest idle time */
						else if(config->priority[worker] ==
							config->priority[curr_workers[index]])
						{
							double worker_idle_time = sched_ctx_hypervisor_get_idle_time(sched_ctx, worker);
							double curr_worker_idle_time = sched_ctx_hypervisor_get_idle_time(sched_ctx, curr_workers[index]);
							if(worker_idle_time > curr_worker_idle_time)
								curr_workers[index] = worker;
						}
					}
				}
			}
		}
			
		if(curr_workers[index] < 0)
		{
			*nworkers = index;
			break;
		}
	}

	if(workers->init_cursor)
		workers->deinit_cursor(workers);

	return curr_workers;
}

static unsigned _get_potential_nworkers(struct policy_config *config, unsigned sched_ctx, enum starpu_archtype arch)
{
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);

	unsigned potential_workers = 0;
	int worker;

	if(workers->init_cursor)
		workers->init_cursor(workers);
	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		enum starpu_archtype curr_arch = starpu_worker_get_type(worker);
                if(arch == 0 || curr_arch == arch)
                {
			if(!config->fixed_workers[worker])
				potential_workers++;
		}
	}
	if(workers->init_cursor)
		workers->deinit_cursor(workers);
	
	return potential_workers;
}

static unsigned _get_nworkers_to_move(unsigned req_sched_ctx)
{
       	struct policy_config *config = sched_ctx_hypervisor_get_config(req_sched_ctx);
	unsigned nworkers = starpu_get_nworkers_of_sched_ctx(req_sched_ctx);
	unsigned nworkers_to_move = 0;
	
	unsigned potential_moving_workers = _get_potential_nworkers(config, req_sched_ctx, 0);
	if(potential_moving_workers > 0)
	{
		if(potential_moving_workers <= config->min_nworkers)
			/* if we have to give more than min better give it all */ 
			/* => empty ctx will block until having the required workers */
			
			nworkers_to_move = potential_moving_workers; 
		else if(potential_moving_workers > config->max_nworkers)
		{
			if((potential_moving_workers - config->granularity) > config->max_nworkers)
				nworkers_to_move = config->granularity;
			else
				nworkers_to_move = potential_moving_workers - config->max_nworkers;
 
		}
		else if(potential_moving_workers > config->granularity)
		{
			if((nworkers - config->granularity) > config->min_nworkers)	
				nworkers_to_move = config->granularity;
			else
				nworkers_to_move = potential_moving_workers - config->min_nworkers;
		}
		else
		{
			int nfixed_workers = nworkers - potential_moving_workers;
			if(nfixed_workers >= config->min_nworkers)
				nworkers_to_move = potential_moving_workers;
			else
				nworkers_to_move = potential_moving_workers - (config->min_nworkers - nfixed_workers);	
		}

		if((nworkers - nworkers_to_move) > config->max_nworkers)
			nworkers_to_move = nworkers - config->max_nworkers;
	}
	return nworkers_to_move;
}

static unsigned _simple_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize)
{
	int ret = 1;
	if(force_resize)
		pthread_mutex_lock(&act_hypervisor_mutex);
	else
		ret = pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{					
		unsigned nworkers_to_move = _get_nworkers_to_move(sender_sched_ctx);

		if(nworkers_to_move > 0)
		{
			unsigned poor_sched_ctx = STARPU_NMAX_SCHED_CTXS;
			if(receiver_sched_ctx == STARPU_NMAX_SCHED_CTXS)
				poor_sched_ctx = _find_poor_sched_ctx(sender_sched_ctx, nworkers_to_move);
			else
			{
				poor_sched_ctx = receiver_sched_ctx;
				struct policy_config *config = sched_ctx_hypervisor_get_config(poor_sched_ctx);
				unsigned nworkers = starpu_get_nworkers_of_sched_ctx(poor_sched_ctx);
				unsigned nshared_workers = starpu_get_nshared_workers(sender_sched_ctx, poor_sched_ctx);
				if((nworkers+nworkers_to_move-nshared_workers) > config->max_nworkers)
					nworkers_to_move = nworkers > config->max_nworkers ? 0 : (config->max_nworkers - nworkers+nshared_workers);
				if(nworkers_to_move == 0) poor_sched_ctx = STARPU_NMAX_SCHED_CTXS;
			}


			if(poor_sched_ctx != STARPU_NMAX_SCHED_CTXS)
			{						
				int *workers_to_move = _get_first_workers(sender_sched_ctx, &nworkers_to_move, 0);
				sched_ctx_hypervisor_move_workers(sender_sched_ctx, poor_sched_ctx, workers_to_move, nworkers_to_move);
				
				struct policy_config *new_config = sched_ctx_hypervisor_get_config(poor_sched_ctx);
				int i;
				for(i = 0; i < nworkers_to_move; i++)
					new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;
				
				free(workers_to_move);
			}
		}	
		pthread_mutex_unlock(&act_hypervisor_mutex);
		return 1;
	}
	return 0;

}

static int* _get_workers_to_move(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int *nworkers)
{
        int *workers = NULL;
        double v_receiver = sched_ctx_hypervisor_get_ctx_velocity(receiver_sched_ctx);
        double receiver_remainig_flops = sched_ctx_hypervisor_get_flops_left(receiver_sched_ctx);
        double sender_exp_end = sched_ctx_hypervisor_get_exp_end(sender_sched_ctx);
        double sender_v_cpu = sched_ctx_hypervisor_get_cpu_velocity(sender_sched_ctx);
//      double v_gcpu = sched_ctx_hypervisor_get_gpu_velocity(sender_sched_ctx);                                                                                                                                                                                                                                                                                                                                                                                                              

        double v_for_rctx = (receiver_remainig_flops/(sender_exp_end - starpu_timing_now())) - v_receiver;
//      v_for_rctx /= 2;                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

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
                                        for(i = 0; i < ngpus; i++)
                                                workers[(*nworkers)++] = gpus[i];

                                        for(i = 0; i < ncpus; i++)
                                                workers[(*nworkers)++] = cpus[i];

                                        free(gpus);
                                        free(cpus);
                                }
                        }
                }
		else
                {
                        int nworkers_to_move = _get_nworkers_to_move(sender_sched_ctx);

                        if(sender_nworkers - nworkers_to_move >= sender_config->min_nworkers)
                        {
                                unsigned nshared_workers = starpu_get_nshared_workers(sender_sched_ctx, receiver_sched_ctx);
                                if((nworkers_ctx + nworkers_to_move - nshared_workers) > config->max_nworkers)
                                        nworkers_to_move = nworkers_ctx > config->max_nworkers ? 0 : (config->max_nworkers - nworkers_ctx + nshared_workers);

                                if(nworkers_to_move > 0)
                                {
                                        workers = _get_first_workers(sender_sched_ctx, &nworkers_to_move, 0);
                                        *nworkers = nworkers_to_move;
                                }
                        }
                }
        }
        return workers;
}

static unsigned _simple_resize2(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize)
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

static unsigned simple_resize(unsigned sender_sched_ctx)
{
	return _simple_resize(sender_sched_ctx, STARPU_NMAX_SCHED_CTXS, 1);
}

static void simple_manage_idle_time(unsigned req_sched_ctx, int worker, double idle_time)
{
       	struct policy_config *config = sched_ctx_hypervisor_get_config(req_sched_ctx);

	if(config != NULL && idle_time > config->max_idle[worker])
		simple_resize(req_sched_ctx);
	return;
}

int _find_fastest_sched_ctx()
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

	double first_exp_end = sched_ctx_hypervisor_get_exp_end(sched_ctxs[0]);
	int fastest_sched_ctx = first_exp_end == -1.0  ? -1 : sched_ctxs[0];
	double curr_exp_end = 0.0;
	int i;
	for(i = 1; i < nsched_ctxs; i++)
	{
		curr_exp_end = sched_ctx_hypervisor_get_exp_end(sched_ctxs[i]);
		if(first_exp_end > curr_exp_end && curr_exp_end != -1.0)
		{
			first_exp_end = curr_exp_end;
			fastest_sched_ctx = sched_ctxs[i];
		}
	}

	return fastest_sched_ctx;

}

int _find_slowest_sched_ctx()
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

	int slowest_sched_ctx = -1;
	double curr_exp_end = 0.0;
	double last_exp_end = -1.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		curr_exp_end = sched_ctx_hypervisor_get_exp_end(sched_ctxs[i]);
		/*if it hasn't started bc of no ressources give it priority */
		if(curr_exp_end == -1.0)
			return sched_ctxs[i];
		if(last_exp_end < curr_exp_end)
		{
			slowest_sched_ctx = sched_ctxs[i];
			last_exp_end = curr_exp_end;
		}
	}

	return slowest_sched_ctx;

}

int _find_slowest_available_sched_ctx(unsigned sched_ctx)
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
			curr_exp_end = sched_ctx_hypervisor_get_exp_end(sched_ctxs[i]);
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

static void simple_manage_gflops_rate(unsigned sched_ctx)
{
	double exp_end = sched_ctx_hypervisor_get_exp_end(sched_ctx);
	double flops_left_pct = sched_ctx_hypervisor_get_flops_left_pct(sched_ctx);

	if(flops_left_pct == 0.0f)
	{
		int slowest_sched_ctx = _find_slowest_available_sched_ctx(sched_ctx);
		if(slowest_sched_ctx != -1)
		{
			double slowest_flops_left_pct = sched_ctx_hypervisor_get_flops_left_pct(slowest_sched_ctx);
			printf("ctx %d finished & gives away the res to %d; slow_left %lf\n", sched_ctx, slowest_sched_ctx, slowest_flops_left_pct);
			if(slowest_flops_left_pct != 0.0f)
			{
				struct policy_config* config = sched_ctx_hypervisor_get_config(sched_ctx);
				config->min_nworkers = 0;
				config->max_nworkers = 0;
				_simple_resize(sched_ctx, slowest_sched_ctx, 1);
				sched_ctx_hypervisor_stop_resize(slowest_sched_ctx);
			}
		}
	}

	int fastest_sched_ctx = _find_fastest_sched_ctx();
	int slowest_sched_ctx = _find_slowest_sched_ctx();
	if(fastest_sched_ctx != -1 && slowest_sched_ctx != -1 && fastest_sched_ctx != slowest_sched_ctx)
	{
		double fastest_exp_end = sched_ctx_hypervisor_get_exp_end(fastest_sched_ctx);
		double slowest_exp_end = sched_ctx_hypervisor_get_exp_end(slowest_sched_ctx);
		double fastest_bef_res_exp_end = sched_ctx_hypervisor_get_bef_res_exp_end(fastest_sched_ctx);
		double slowest_bef_res_exp_end = sched_ctx_hypervisor_get_bef_res_exp_end(slowest_sched_ctx);
//					       (fastest_bef_res_exp_end < slowest_bef_res_exp_end || 
//						fastest_bef_res_exp_end == 0.0 || slowest_bef_res_exp_end == 0)))
		
		if((slowest_exp_end == -1.0 && fastest_exp_end != -1.0) || ((fastest_exp_end + (fastest_exp_end*0.5)) < slowest_exp_end ))
		{
			double fast_flops_left_pct = sched_ctx_hypervisor_get_flops_left_pct(fastest_sched_ctx);
			if(fast_flops_left_pct < 0.8)
				_simple_resize(fastest_sched_ctx, slowest_sched_ctx, 0);
		}
	}
}


struct hypervisor_policy idle_policy = {
	.manage_idle_time = simple_manage_idle_time,
	.manage_gflops_rate = simple_manage_gflops_rate,
	.resize = simple_resize,
};

struct hypervisor_policy app_driven_policy = {
	.manage_idle_time = simple_manage_idle_time,
	.manage_gflops_rate = simple_manage_gflops_rate,
	.resize = simple_resize,
};

struct hypervisor_policy gflops_rate_policy = {
	.manage_idle_time = simple_manage_idle_time,
	.manage_gflops_rate = simple_manage_gflops_rate,
	.resize = simple_resize,
};
