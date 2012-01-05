#include <sched_ctx_hypervisor_intern.h>

unsigned imposed_resize = 0;
struct starpu_sched_ctx_hypervisor_criteria* criteria = NULL;

extern struct hypervisor_policy simple_policy;

static void idle_time_cb(unsigned sched_ctx, int worker, double idle_time);
static void pushed_task_cb(unsigned sched_ctx, int worker);
static void poped_task_cb(unsigned sched_ctx, int worker);
static void post_exec_hook_cb(unsigned sched_ctx, int taskid);
static void reset_idle_time_cb(unsigned sched_ctx, int  worker);

static void _load_hypervisor_policy(int type)
{
	switch(type)
	{
	case SIMPLE_POLICY:
		hypervisor.policy.init = simple_policy.init;
		hypervisor.policy.deinit = simple_policy.deinit;
		hypervisor.policy.add_sched_ctx = simple_policy.add_sched_ctx;
		hypervisor.policy.remove_sched_ctx = simple_policy.remove_sched_ctx;
		hypervisor.policy.ioctl = simple_policy.ioctl;
		hypervisor.policy.manage_idle_time = simple_policy.manage_idle_time;
		hypervisor.policy.update_config = simple_policy.update_config;
		hypervisor.policy.resize = simple_policy.resize;
		hypervisor.policy.manage_task_flux = simple_policy.manage_task_flux;
		break;
	}
}

struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(int type)
{
	hypervisor.min_tasks = 0;
	hypervisor.nsched_ctxs = 0;
	pthread_mutex_init(&act_hypervisor_mutex, NULL);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		hypervisor.resize[i] = 0;
		hypervisor.configurations[i] = NULL;
		hypervisor.sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].sched_ctx = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].config = NULL;
		hypervisor.sched_ctx_w[i].temp_npushed_tasks = 0;
		hypervisor.sched_ctx_w[i].temp_npoped_tasks = 0;

		int j;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
		{
			hypervisor.sched_ctx_w[i].current_idle_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].pushed_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].poped_tasks[j] = 0;

		}
	}

	_load_hypervisor_policy(type);

	criteria = (struct starpu_sched_ctx_hypervisor_criteria*)malloc(sizeof(struct starpu_sched_ctx_hypervisor_criteria));
	criteria->idle_time_cb = idle_time_cb;
	criteria->pushed_task_cb = pushed_task_cb;
	criteria->poped_task_cb = poped_task_cb;
	criteria->post_exec_hook_cb = post_exec_hook_cb;
	criteria->reset_idle_time_cb = reset_idle_time_cb;
	return criteria;
}

void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx)
{
	imposed_resize = 1;
	hypervisor.resize[sched_ctx] = 0;
}

void sched_ctx_hypervisor_start_resize(unsigned sched_ctx)
{
	imposed_resize = 1;
	hypervisor.resize[sched_ctx] = 1;
}

void sched_ctx_hypervisor_shutdown(void)
{
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		sched_ctx_hypervisor_stop_resize(hypervisor.sched_ctxs[i]);
                if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && hypervisor.nsched_ctxs > 0)
			sched_ctx_hypervisor_ignore_ctx(i);
	}
	free(criteria);
	pthread_mutex_destroy(&act_hypervisor_mutex);
}

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx)
{	
	hypervisor.configurations[sched_ctx] = (struct starpu_htbl32_node_s*)malloc(sizeof(struct starpu_htbl32_node_s));
	hypervisor.steal_requests[sched_ctx] = (struct starpu_htbl32_node_s*)malloc(sizeof(struct starpu_htbl32_node_s));
	hypervisor.resize_requests[sched_ctx] = (struct starpu_htbl32_node_s*)malloc(sizeof(struct starpu_htbl32_node_s));

	hypervisor.policy.add_sched_ctx(sched_ctx);
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = sched_ctx;
	hypervisor.sched_ctxs[hypervisor.nsched_ctxs++] = sched_ctx;
}

static int _get_first_free_sched_ctx(int *sched_ctxs, unsigned nsched_ctxs)
{
        int i;
        for(i = 0; i < nsched_ctxs; i++)
                if(sched_ctxs[i] == STARPU_NMAX_SCHED_CTXS)
                        return i;

        return STARPU_NMAX_SCHED_CTXS;
}

/* rearange array of sched_ctxs in order not to have {MAXVAL, MAXVAL, 5, MAXVAL, 7}    
   and have instead {5, 7, MAXVAL, MAXVAL, MAXVAL}                                    
   it is easier afterwards to iterate the array                           
*/
static void _rearange_sched_ctxs(int *sched_ctxs, int old_nsched_ctxs)
{
        int first_free_id = STARPU_NMAX_SCHED_CTXS;
        int i;
        for(i = 0; i < old_nsched_ctxs; i++)
        {
                if(sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS)
                {
                        first_free_id = _get_first_free_sched_ctx(sched_ctxs, old_nsched_ctxs);
                        if(first_free_id != STARPU_NMAX_SCHED_CTXS)
			{
                                sched_ctxs[first_free_id] = sched_ctxs[i];
				sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
			}
                }
	}
}

void sched_ctx_hypervisor_ignore_ctx(unsigned sched_ctx)
{
        unsigned i;
        for(i = 0; i < hypervisor.nsched_ctxs; i++)
        {
                if(hypervisor.sched_ctxs[i] == sched_ctx)
                {
                        hypervisor.sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
			break;
                }
        }

        _rearange_sched_ctxs(hypervisor.sched_ctxs, hypervisor.nsched_ctxs);
	hypervisor.nsched_ctxs--;
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = STARPU_NMAX_SCHED_CTXS;
	hypervisor.policy.remove_sched_ctx(sched_ctx);

	free(hypervisor.configurations[sched_ctx]);
	free(hypervisor.steal_requests[sched_ctx]);
	free(hypervisor.resize_requests[sched_ctx]);
}

void sched_ctx_hypervisor_set_config(unsigned sched_ctx, void *config)
{
	    printf("%d: ", sched_ctx );
	if(hypervisor.sched_ctx_w[sched_ctx].config != NULL && config != NULL)
	  {
		hypervisor.policy.update_config(hypervisor.sched_ctx_w[sched_ctx].config, config);
	  }
	else
		hypervisor.sched_ctx_w[sched_ctx].config = config;

	return;
}

void* sched_ctx_hypervisor_get_config(unsigned sched_ctx)
{
	return hypervisor.sched_ctx_w[sched_ctx].config;
}

void sched_ctx_hypervisor_ioctl(unsigned sched_ctx, ...)
{
	va_list varg_list;
	va_start(varg_list, sched_ctx);

	int arg_type;
	int stop = 0;
	int task_tag = -1;

	while ((arg_type = va_arg(varg_list, int)) != 0) 
	{
		switch(arg_type)
		{
		case HYPERVISOR_TIME_TO_APPLY:
			task_tag = va_arg(varg_list, int);
			stop = 1;
			break;

		case HYPERVISOR_MIN_TASKS:
			hypervisor.min_tasks = va_arg(varg_list, int);
			break;

		}
		if(stop) break;
	}

	va_end(varg_list);
	va_start(varg_list, sched_ctx);

	/* if config not null => save hypervisor configuration and consider it later */
	void *config = hypervisor.policy.ioctl(sched_ctx, varg_list, (task_tag > 0));
	if(config != NULL)
		_starpu_htbl_insert_32(&hypervisor.configurations[sched_ctx], (uint32_t)task_tag, config);

	return;
}

static int get_ntasks( int *tasks)
{
	int ntasks = 0;
	int j;
	for(j = 0; j < STARPU_NMAXWORKERS; j++)
	{
		ntasks += tasks[j];
	}
	return ntasks;
}

static void reset_ntasks( int *tasks)
{
	int j;
	for(j = 0; j < STARPU_NMAXWORKERS; j++)
	{
		tasks[j] = 0;
	}
	return;
}

static unsigned check_tasks_of_sched_ctx(unsigned sched_ctx)
{
	int ntasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].pushed_tasks);
	
	return ntasks > hypervisor.min_tasks;
}

void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move)
{
	if(hypervisor.resize[sender_sched_ctx])
	{
		int j;
		printf("resize ctx %d with", sender_sched_ctx);
		for(j = 0; j < nworkers_to_move; j++)
			printf(" %d", workers_to_move[j]);
		printf("\n");

		starpu_remove_workers_from_sched_ctx(workers_to_move, nworkers_to_move, sender_sched_ctx);
		starpu_add_workers_to_sched_ctx(workers_to_move, nworkers_to_move, receiver_sched_ctx);

		int i;
		for(i = 0; i < nworkers_to_move; i++)
			hypervisor.sched_ctx_w[sender_sched_ctx].current_idle_time[workers_to_move[i]] = 0.0;
	}

	return;
}

unsigned sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag)
{
	if(task_tag == -1)
	{
		return hypervisor.policy.resize(sched_ctx, hypervisor.sched_ctxs, hypervisor.nsched_ctxs);
	}
	else
	{	
		unsigned *sched_ctx_pt = (unsigned*)malloc(sizeof(unsigned));
		*sched_ctx_pt = sched_ctx;
		_starpu_htbl_insert_32(&hypervisor.resize_requests[sched_ctx], (uint32_t)task_tag, (void*)sched_ctx_pt);	
		return 0;
	}
}

void get_overage_workers(unsigned sched_ctx, int *workerids, int nworkers, int *overage_workers, int *noverage_workers)
{
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);
	int worker, i, found = -1;

	if(workers->init_cursor)
		workers->init_cursor(workers);

	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		for(i = 0; i < nworkers; i++)
			if(workerids[i] == worker)
			{
				found = worker;
				break;
			}
		if(found == -1)
			overage_workers[(*noverage_workers)++]  = worker;
		found = -1;
	}

	if(workers->init_cursor)
		workers->deinit_cursor(workers);
}

void sched_ctx_hypervisor_steal_workers(unsigned sched_ctx, int *workerids, int nworkers, int task_tag)
{
	/* do it right now */
	if(task_tag == -1)	
	{
		pthread_mutex_lock(&act_hypervisor_mutex);
		
		if(hypervisor.sched_ctx_w[sched_ctx].sched_ctx != STARPU_NMAX_SCHED_CTXS)
		{
			printf("do request\n");

			int overage_workers[STARPU_NMAXWORKERS];
			int noverage_workers = 0;
			get_overage_workers(sched_ctx, workerids, nworkers, overage_workers, &noverage_workers);
			starpu_add_workers_to_sched_ctx(workerids, nworkers, sched_ctx);

			sched_ctx_hypervisor_ioctl(sched_ctx, 
						   HYPERVISOR_PRIORITY, workerids, nworkers, 1,
						   NULL);		

			if(noverage_workers > 0)
				starpu_remove_workers_from_sched_ctx(overage_workers, noverage_workers, sched_ctx);
			
			int i;
			for(i = 0; i < hypervisor.nsched_ctxs; i++)
				if(hypervisor.sched_ctxs[i] != sched_ctx && hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS)
					starpu_remove_workers_from_sched_ctx(workerids, nworkers, hypervisor.sched_ctxs[i]);
		}
		
		pthread_mutex_unlock(&act_hypervisor_mutex);
	}
	else
	{
		struct sched_ctx_hypervisor_adjustment* adjustment = (struct sched_ctx_hypervisor_adjustment*)malloc(sizeof(struct sched_ctx_hypervisor_adjustment));
		int i;
		for(i = 0; i < nworkers; i++)
			adjustment->workerids[i] = workerids[i];
		adjustment->nworkers = nworkers;
		
		_starpu_htbl_insert_32(&hypervisor.steal_requests[sched_ctx], (uint32_t)task_tag, (void*)adjustment);	
	}

	return ;
}

static void reset_idle_time_cb(unsigned sched_ctx, int worker)
{
	if(hypervisor.resize[sched_ctx])
		hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker] = 0.0;
}

static void idle_time_cb(unsigned sched_ctx, int worker, double idle_time)
{
	if(hypervisor.resize[sched_ctx] && hypervisor.nsched_ctxs > 1 && hypervisor.policy.manage_idle_time)
	{
		hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker] += idle_time;
		hypervisor.policy.manage_idle_time(sched_ctx, worker, hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker]);

	}
	return;
}

static void working_time_cb(unsigned sched_ctx, int worker, double working_time, unsigned current_nworkers)
{
	return;
}

int* sched_ctx_hypervisor_get_sched_ctxs()
{
	return hypervisor.sched_ctxs;
}

int sched_ctx_hypervisor_get_nsched_ctxs()
{
	return hypervisor.nsched_ctxs;
}

double sched_ctx_hypervisor_get_debit(unsigned sched_ctx)
{
	unsigned nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx);
	if(nworkers == 0)
		return 0.0;

	int npushed_tasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].pushed_tasks);
	int npoped_tasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].poped_tasks);
	if(hypervisor.sched_ctx_w[sched_ctx].temp_npushed_tasks != npushed_tasks || hypervisor.sched_ctx_w[sched_ctx].temp_npoped_tasks!= npoped_tasks)
	{
		hypervisor.sched_ctx_w[sched_ctx].temp_npushed_tasks = npushed_tasks;
		hypervisor.sched_ctx_w[sched_ctx].temp_npoped_tasks = npoped_tasks;
		
		STARPU_ASSERT(npoped_tasks <= npushed_tasks);
		if(npushed_tasks > 0 && npoped_tasks > 0)
		{
			double debit = (((double)npoped_tasks)*1.0)/((double)npushed_tasks * 1.0);
			return debit;
		}
	}

	return 0.0;
}

static void pushed_task_cb(unsigned sched_ctx, int worker)
{	
	hypervisor.sched_ctx_w[sched_ctx].pushed_tasks[worker]++;
       
	int ntasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].pushed_tasks);
	
	if(!imposed_resize)
		hypervisor.resize[sched_ctx] = (ntasks > hypervisor.min_tasks);
}

static void poped_task_cb(unsigned sched_ctx, int worker)
{
	hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker]++;
	
	/* if(hypervisor.nsched_ctxs > 1) */
	/* 	hypervisor.policy.manage_task_flux(sched_ctx); */
}

static void post_exec_hook_cb(unsigned sched_ctx, int task_tag)
{
	STARPU_ASSERT(task_tag > 0);

	if(hypervisor.nsched_ctxs > 1 && hypervisor.resize[sched_ctx])
	{
		unsigned conf_sched_ctx;
		int i;
		for(i = 0; i < hypervisor.nsched_ctxs; i++)
		{
			conf_sched_ctx = hypervisor.sched_ctxs[i];
			void *config = _starpu_htbl_search_32(hypervisor.configurations[conf_sched_ctx], (uint32_t)task_tag);
			if(config && config != hypervisor.configurations[conf_sched_ctx])
			{
				sched_ctx_hypervisor_set_config(conf_sched_ctx, config);
				free(config);
				_starpu_htbl_insert_32(&hypervisor.configurations[sched_ctx], (uint32_t)task_tag, NULL);
			}
		}	
		
		struct sched_ctx_hypervisor_adjustment *adjustment = (struct sched_ctx_hypervisor_adjustment*) _starpu_htbl_search_32(hypervisor.steal_requests[sched_ctx], (uint32_t)task_tag);
		if(adjustment && adjustment != hypervisor.steal_requests[sched_ctx])
		{
			sched_ctx_hypervisor_steal_workers(sched_ctx, adjustment->workerids, adjustment->nworkers, -1);
			free(adjustment);
			_starpu_htbl_insert_32(&hypervisor.steal_requests[sched_ctx], (uint32_t)task_tag, NULL);
		}
		
		unsigned *sched_ctx_pt = _starpu_htbl_search_32(hypervisor.resize_requests[sched_ctx], (uint32_t)task_tag);
		if(sched_ctx_pt && sched_ctx_pt != hypervisor.resize_requests[sched_ctx])
		{
			sched_ctx_hypervisor_resize(sched_ctx, -1);
			free(sched_ctx_pt);
			_starpu_htbl_insert_32(&hypervisor.resize_requests[sched_ctx], (uint32_t)task_tag, NULL);
		}
	}
}
