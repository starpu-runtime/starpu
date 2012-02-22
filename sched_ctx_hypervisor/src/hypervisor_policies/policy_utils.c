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

unsigned _find_poor_sched_ctx(unsigned req_sched_ctx, int nworkers_to_move)
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
	struct sched_ctx_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
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
			if(arch == -1 || curr_arch == arch)
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
							double worker_idle_time = sc_w->current_idle_time[worker];
							double curr_worker_idle_time = sc_w->current_idle_time[curr_workers[index]];
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

unsigned _get_potential_nworkers(struct policy_config *config, unsigned sched_ctx, enum starpu_archtype arch)
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

unsigned _get_nworkers_to_move(unsigned req_sched_ctx)
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
//				nworkers_to_move = config->granularity;
				nworkers_to_move = potential_moving_workers;
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

unsigned _resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize)
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
				int *workers_to_move = _get_first_workers(sender_sched_ctx, &nworkers_to_move, -1);
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


unsigned _resize_to_unknown_receiver(unsigned sender_sched_ctx)
{
	return _resize(sender_sched_ctx, STARPU_NMAX_SCHED_CTXS, 0);
}

