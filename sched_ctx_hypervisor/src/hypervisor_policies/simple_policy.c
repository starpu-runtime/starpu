#include <sched_ctx_hypervisor.h>
#include <pthread.h>

#define MAX_IDLE_TIME 5000000000
#define MIN_WORKING_TIME 500

struct simple_policy_config {
	/* underneath this limit we cannot resize */
	unsigned min_nprocs;

	/* above this limit we cannot resize */
	unsigned max_nprocs;
	
	/*resize granularity */
	unsigned granularity;

	/* priority for a worker to stay in this context */
	/* the smaller the priority the faster it will be moved */
	/* to another context */
	int priority[STARPU_NMAXWORKERS];

	/* above this limit the priority of the worker is reduced */
	double max_idle[STARPU_NMAXWORKERS];

	/* underneath this limit the priority of the worker is reduced */
	double min_working[STARPU_NMAXWORKERS];

	/* workers that will not move */
	unsigned fixed_procs[STARPU_NMAXWORKERS];

	/* max idle for the workers that will be added during the resizing process*/
	double new_workers_max_idle;
};

static struct simple_policy_config* _create_config(void)
{
	struct simple_policy_config *config = (struct simple_policy_config *)malloc(sizeof(struct simple_policy_config));
	config->min_nprocs = 0;
	config->max_nprocs = 0;	
	config->new_workers_max_idle = MAX_IDLE_TIME;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		config->granularity = 1;
		config->priority[i] = 0;
		config->fixed_procs[i] = 0;
		config->max_idle[i] = MAX_IDLE_TIME;
		config->min_working[i] = MIN_WORKING_TIME;
	}
	
	return config;
}

static void simple_add_sched_ctx(unsigned sched_ctx)
{
	struct simple_policy_config *config = _create_config();
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

static unsigned _get_highest_priority_sched_ctx(unsigned req_sched_ctx, int *sched_ctxs, int nsched_ctxs)
{
	int i;
	int highest_priority = -1;
	int current_priority = 0;
	unsigned sched_ctx = STARPU_NMAX_SCHED_CTXS;

	for(i = 0; i < nsched_ctxs; i++)
	{
		if(sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && sched_ctxs[i] != req_sched_ctx)
		{
			current_priority = _compute_priority(sched_ctxs[i]);
			if (highest_priority < current_priority)
			{
				highest_priority = current_priority;
				sched_ctx = sched_ctxs[i];
			}
		}
	}
	
	return sched_ctx;
}

int* _get_first_workers(unsigned sched_ctx, int nworkers)
{
	struct simple_policy_config *config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(sched_ctx);

	int *curr_workers = (int*)malloc(nworkers * sizeof(int));
	int i;
	for(i = 0; i < nworkers; i++)
		curr_workers[i] = -1;

	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);
	int index;
	int worker;
	int considered = 0;

	if(workers->init_cursor)
		workers->init_cursor(workers);

	for(index = 0; index < nworkers; index++)
	{
		while(workers->has_next(workers))
		{
			considered = 0;
			worker = workers->get_next(workers);
			if(!config->fixed_procs[worker])
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
			break;
	}

	if(workers->init_cursor)
		workers->deinit_cursor(workers);

	return curr_workers;
}

static int _get_potential_nworkers(struct simple_policy_config *config, unsigned sched_ctx)
{
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);

	int potential_workers = 0;
	int worker;

	if(workers->init_cursor)
		workers->init_cursor(workers);
	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		if(!config->fixed_procs[worker])
			potential_workers++;
	}
	if(workers->init_cursor)
		workers->deinit_cursor(workers);
	
	return potential_workers;
}

static unsigned simple_manage_idle_time(unsigned req_sched_ctx, int *sched_ctxs, int nsched_ctxs, int worker, double idle_time)
{
       	struct simple_policy_config *config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(req_sched_ctx);

	if(config != NULL && idle_time > config->max_idle[worker])
	{
		int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{					
			
			unsigned nworkers = starpu_get_nworkers_of_sched_ctx(req_sched_ctx);
			unsigned nworkers_to_move = 0;
			
			/* leave at least one */
			int potential_moving_workers = _get_potential_nworkers(config, req_sched_ctx);
			if(potential_moving_workers > 0)
			{
				if(potential_moving_workers > config->granularity)
				{
					if((nworkers - config->granularity) > config->min_nprocs)	
						nworkers_to_move = config->granularity;
				}
				else
				{
					int nfixed_workers = nworkers - potential_moving_workers;
					if(nfixed_workers >= config->min_nprocs)
						nworkers_to_move = potential_moving_workers;
					else
						nworkers_to_move = potential_moving_workers - (config->min_nprocs - nfixed_workers);	
				}
			}

			if(nworkers_to_move > 0)
			{
				unsigned prio_sched_ctx = _get_highest_priority_sched_ctx(req_sched_ctx, sched_ctxs, nsched_ctxs);
				if(prio_sched_ctx != STARPU_NMAX_SCHED_CTXS)
				{					
					int *workers_to_move = _get_first_workers(req_sched_ctx, nworkers_to_move);
					sched_ctx_hypervisor_resize(req_sched_ctx, prio_sched_ctx, workers_to_move, nworkers_to_move);

					struct simple_policy_config *prio_config = (struct simple_policy_config*)sched_ctx_hypervisor_get_config(prio_sched_ctx);
					int i;
					for(i = 0; i < nworkers_to_move; i++)
						prio_config->max_idle[workers_to_move[i]] = prio_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? prio_config->max_idle[workers_to_move[i]] :  prio_config->new_workers_max_idle;
					
					free(workers_to_move);
				}
			}	
			pthread_mutex_unlock(&act_hypervisor_mutex);
			return 0;
		}
	}
	return 1;
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

		case HYPERVISOR_MIN_PROCS:
			config->min_nprocs = va_arg(varg_list, unsigned);
			break;

		case HYPERVISOR_MAX_PROCS:
			config->max_nprocs = va_arg(varg_list, unsigned);
			break;

		case HYPERVISOR_GRANULARITY:
			config->granularity = va_arg(varg_list, unsigned);
			break;

		case HYPERVISOR_FIXED_PROCS:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);

			for(i = 0; i < nworkers; i++)
				config->fixed_procs[workerids[i]] = 1;
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

	old->min_nprocs = new->min_nprocs != 0 ? new->min_nprocs : old->min_nprocs ;
	old->max_nprocs = new->max_nprocs != 0 ? new->max_nprocs : old->max_nprocs ;
	old->new_workers_max_idle = new->new_workers_max_idle != MAX_IDLE_TIME ? new->new_workers_max_idle : old->new_workers_max_idle;
	old->granularity = new->min_nprocs != 1 ? new->granularity : old->granularity;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		old->priority[i] = new->priority[i] != 0 ? new->priority[i] : old->priority[i];
		old->fixed_procs[i] = new->fixed_procs[i] != 0 ? new->fixed_procs[i] : old->fixed_procs[i];
		old->max_idle[i] = new->max_idle[i] != MAX_IDLE_TIME ? new->max_idle[i] : old->max_idle[i];;
		old->min_working[i] = new->min_working[i] != MIN_WORKING_TIME ? new->min_working[i] : old->min_working[i];
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
	.update_config = simple_update_config
};
