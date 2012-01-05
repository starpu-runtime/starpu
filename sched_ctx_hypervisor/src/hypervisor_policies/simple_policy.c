#include <sched_ctx_hypervisor.h>
#include <pthread.h>

#define MAX_IDLE_TIME 5000000000
#define MIN_WORKING_TIME 500

struct simple_policy_config {
	/* underneath this limit we cannot resize */
	int min_nworkers;

	/* above this limit we cannot resize */
	int max_nworkers;
	
	/*resize granularity */
	int granularity;

	/* priority for a worker to stay in this context */
	/* the smaller the priority the faster it will be moved */
	/* to another context */
	int priority[STARPU_NMAXWORKERS];

	/* above this limit the priority of the worker is reduced */
	double max_idle[STARPU_NMAXWORKERS];

	/* underneath this limit the priority of the worker is reduced */
	double min_working[STARPU_NMAXWORKERS];

	/* workers that will not move */
	int fixed_workers[STARPU_NMAXWORKERS];

	/* max idle for the workers that will be added during the resizing process*/
	double new_workers_max_idle;

	/* above this context we allow removing all workers */
	double empty_ctx_max_idle[STARPU_NMAXWORKERS];
};

static struct simple_policy_config* _create_config(void)
{
	struct simple_policy_config *config = (struct simple_policy_config *)malloc(sizeof(struct simple_policy_config));
	config->min_nworkers = -1;
	config->max_nworkers = -1;	
	config->new_workers_max_idle = -1.0;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		config->granularity = -1;
		config->priority[i] = -1;
		config->fixed_workers[i] = -1;
		config->max_idle[i] = -1.0;
		config->empty_ctx_max_idle[i] = -1.0;
		config->min_working[i] = -1.0;
	}
	
	return config;
}

static void simple_add_sched_ctx(unsigned sched_ctx)
{
	struct simple_policy_config *config = _create_config();
	config->min_nworkers = 0;
	config->max_nworkers = 0;	
	config->new_workers_max_idle = MAX_IDLE_TIME;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		config->granularity = 1;
		config->priority[i] = 0;
		config->fixed_workers[i] = 0;
		config->max_idle[i] = MAX_IDLE_TIME;
		config->empty_ctx_max_idle[i] = MAX_IDLE_TIME;
		config->min_working[i] = MIN_WORKING_TIME;
	}

	sched_ctx_hypervisor_set_config(sched_ctx, config);
}

static int _compute_priority(unsigned sched_ctx)
{
	struct simple_policy_config *config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(sched_ctx);

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


	struct simple_policy_config *config = NULL;

	for(i = 0; i < nsched_ctxs; i++)
	{
		if(sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && sched_ctxs[i] != req_sched_ctx)
		{
			unsigned nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctxs[i]);
			config  = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(sched_ctxs[i]);
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

int* _get_first_workers(unsigned sched_ctx, unsigned *nworkers)
{
	struct simple_policy_config *config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(sched_ctx);

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
				
				if(!considered && (curr_workers[index] < 0 || 
						   config->priority[worker] <
						   config->priority[curr_workers[index]]))
					curr_workers[index] = worker;
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

static unsigned _get_potential_nworkers(struct simple_policy_config *config, unsigned sched_ctx)
{
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);

	unsigned potential_workers = 0;
	int worker;

	if(workers->init_cursor)
		workers->init_cursor(workers);
	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		if(!config->fixed_workers[worker])
			potential_workers++;
	}
	if(workers->init_cursor)
		workers->deinit_cursor(workers);
	
	return potential_workers;
}

static unsigned _get_nworkers_to_move(unsigned req_sched_ctx)
{
       	struct simple_policy_config *config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(req_sched_ctx);
	unsigned nworkers = starpu_get_nworkers_of_sched_ctx(req_sched_ctx);
	unsigned nworkers_to_move = 0;
	
	unsigned potential_moving_workers = _get_potential_nworkers(config, req_sched_ctx);
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
		printf("nworkers = %d nworkers_to_move = %d max_nworkers=%d\n", nworkers, nworkers_to_move, config->max_nworkers);
		if((nworkers - nworkers_to_move) > config->max_nworkers)
			nworkers_to_move = nworkers - config->max_nworkers;
	}
	return nworkers_to_move;
}

static int _find_fastest_sched_ctx()
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

	int fastest_sched_ctx = -1;
	double fastest_debit = -1.0, curr_debit = 0.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		curr_debit = sched_ctx_hypervisor_get_debit(sched_ctxs[i]);
		if(fastest_debit <= curr_debit)
		{
			fastest_debit = curr_debit;
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
	double slowest_debit = 1.0, curr_debit = 0.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		curr_debit = sched_ctx_hypervisor_get_debit(sched_ctxs[i]);
		if(slowest_debit >= curr_debit)
		{
			slowest_debit = curr_debit;
			slowest_sched_ctx = sched_ctxs[i];
		}
	}

	return slowest_sched_ctx;
}

static unsigned _simple_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx)
{
	int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
	if(ret != EBUSY)
	{					
		unsigned nworkers_to_move = _get_nworkers_to_move(sender_sched_ctx);
		
		if(sender_sched_ctx == 2)
			printf("try to resize with nworkers = %d\n", nworkers_to_move);
		if(nworkers_to_move > 0)
		{
			unsigned poor_sched_ctx = STARPU_NMAX_SCHED_CTXS;
			if(receiver_sched_ctx == STARPU_NMAX_SCHED_CTXS)
				poor_sched_ctx = _find_poor_sched_ctx(sender_sched_ctx, nworkers_to_move);
			else
			{
				poor_sched_ctx = receiver_sched_ctx;
				struct simple_policy_config *config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(poor_sched_ctx);
				unsigned nworkers = starpu_get_nworkers_of_sched_ctx(poor_sched_ctx);
				if((nworkers+nworkers_to_move) > config->max_nworkers)
					nworkers_to_move = nworkers > config->max_nworkers ? 0 : (config->max_nworkers - nworkers);
				if(nworkers_to_move == 0) poor_sched_ctx = STARPU_NMAX_SCHED_CTXS;
			}
			
			if(poor_sched_ctx != STARPU_NMAX_SCHED_CTXS)
			{						
				int *workers_to_move = _get_first_workers(sender_sched_ctx, &nworkers_to_move);
				sched_ctx_hypervisor_move_workers(sender_sched_ctx, poor_sched_ctx, workers_to_move, nworkers_to_move);
				
				struct simple_policy_config *new_config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(poor_sched_ctx);
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

static unsigned simple_resize(unsigned sender_sched_ctx)
{
	return _simple_resize(sender_sched_ctx, STARPU_NMAX_SCHED_CTXS);
}

static void simple_manage_idle_time(unsigned req_sched_ctx, int worker, double idle_time)
{
       	struct simple_policy_config *config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(req_sched_ctx);

	if(config != NULL && idle_time > config->max_idle[worker])
		simple_resize(req_sched_ctx);
	return;
}

static void simple_manage_task_flux(unsigned curr_sched_ctx)
{
	double curr_debit = sched_ctx_hypervisor_get_debit(curr_sched_ctx);
	
	int slow_sched_ctx = _find_slowest_sched_ctx();
	int fast_sched_ctx = _find_fastest_sched_ctx();
	if(slow_sched_ctx != fast_sched_ctx && slow_sched_ctx != -1 && fast_sched_ctx != -1)
	{
		if(curr_sched_ctx == slow_sched_ctx)
		{
			double debit_fast = sched_ctx_hypervisor_get_debit(fast_sched_ctx);
			/* only if there is a difference of 30 % */
			if(debit_fast != 0.0 && debit_fast > (curr_debit + curr_debit * 0.1))
				_simple_resize(fast_sched_ctx, curr_sched_ctx);
		}
		
		if(curr_sched_ctx == fast_sched_ctx)
		{
			double debit_slow = sched_ctx_hypervisor_get_debit(slow_sched_ctx);
			/* only if there is a difference of 30 % */
			if(curr_debit != 0.0 && (debit_slow + debit_slow *0.1) < curr_debit)
				_simple_resize(curr_sched_ctx, slow_sched_ctx);
		}
	}
}

static void* simple_ioctl(unsigned sched_ctx, va_list varg_list, unsigned later)
{
	struct simple_policy_config *config = NULL;

	if(later)
		config = _create_config();
	else
		config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(sched_ctx);

	assert(config != NULL);

	int arg_type;
	int i;
	int *workerids;
	int nworkers;

	while ((arg_type = va_arg(varg_list, int)) != 0) 
	{
		switch(arg_type)
		{
		case HYPERVISOR_MAX_IDLE:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			double max_idle = va_arg(varg_list, double);
			
			for(i = 0; i < nworkers; i++)
				config->max_idle[workerids[i]] = max_idle;

			break;

		case HYPERVISOR_EMPTY_CTX_MAX_IDLE:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			double empty_ctx_max_idle = va_arg(varg_list, double);
			
			for(i = 0; i < nworkers; i++)
				config->empty_ctx_max_idle[workerids[i]] = empty_ctx_max_idle;

			break;

		case HYPERVISOR_MIN_WORKING:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			double min_working = va_arg(varg_list, double);

			for(i = 0; i < nworkers; i++)
				config->min_working[workerids[i]] = min_working;

			break;

		case HYPERVISOR_PRIORITY:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			int priority = va_arg(varg_list, int);
	
			for(i = 0; i < nworkers; i++)
				config->priority[workerids[i]] = priority;
			break;

		case HYPERVISOR_MIN_WORKERS:
			config->min_nworkers = va_arg(varg_list, unsigned);
			break;

		case HYPERVISOR_MAX_WORKERS:
			config->max_nworkers = va_arg(varg_list, unsigned);
			if(config->max_nworkers == 0)
			  printf("%d: max nworkers = 0\n", sched_ctx);
			break;

		case HYPERVISOR_GRANULARITY:
			config->granularity = va_arg(varg_list, unsigned);
			break;

		case HYPERVISOR_FIXED_WORKERS:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);

			for(i = 0; i < nworkers; i++)
				config->fixed_workers[workerids[i]] = 1;
			break;

		case HYPERVISOR_NEW_WORKERS_MAX_IDLE:
			config->new_workers_max_idle = va_arg(varg_list, double);
			break;

/* not important for the strateg, needed just to jump these args in the iteration of the args */			
		case HYPERVISOR_TIME_TO_APPLY:
			va_arg(varg_list, int);
			break;

		case HYPERVISOR_MIN_TASKS:
			va_arg(varg_list, int);
			break;

		}
	}

	va_end(varg_list);

	return later ? (void*)config : NULL;
}

static void simple_update_config(void *old_config, void* config)
{
	struct simple_policy_config *old = (struct simple_policy_config*)old_config;
	struct simple_policy_config *new = (struct simple_policy_config*)config;

	printf("new = %d old = %d\n", new->max_nworkers, old->max_nworkers);
	old->min_nworkers = new->min_nworkers != -1 ? new->min_nworkers : old->min_nworkers ;
	old->max_nworkers = new->max_nworkers != -1 ? new->max_nworkers : old->max_nworkers ;
	old->new_workers_max_idle = new->new_workers_max_idle != -1.0 ? new->new_workers_max_idle : old->new_workers_max_idle;
	old->granularity = new->granularity != -1 ? new->granularity : old->granularity;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		old->priority[i] = new->priority[i] != -1 ? new->priority[i] : old->priority[i];
		old->fixed_workers[i] = new->fixed_workers[i] != -1 ? new->fixed_workers[i] : old->fixed_workers[i];
		old->max_idle[i] = new->max_idle[i] != -1.0 ? new->max_idle[i] : old->max_idle[i];
		old->empty_ctx_max_idle[i] = new->empty_ctx_max_idle[i] != -1.0 ? new->empty_ctx_max_idle[i] : old->empty_ctx_max_idle[i];
		old->min_working[i] = new->min_working[i] != -1.0 ? new->min_working[i] : old->min_working[i];
	}
}

static void simple_remove_sched_ctx(unsigned sched_ctx)
{
	sched_ctx_hypervisor_set_config(sched_ctx, NULL);
}

struct hypervisor_policy simple_policy = {
	.init = NULL,
	.deinit = NULL,
	.add_sched_ctx = simple_add_sched_ctx,
	.remove_sched_ctx = simple_remove_sched_ctx,
	.ioctl = simple_ioctl,
	.manage_idle_time = simple_manage_idle_time,
	.manage_task_flux = simple_manage_task_flux,
	.resize = simple_resize,
	.update_config = simple_update_config
};
