#include <sched_ctx_hypervisor_intern.h>

struct starpu_sched_ctx_hypervisor_criteria* criteria = NULL;

extern struct hypervisor_policy simple_policy;

static void idle_time_cb(unsigned sched_ctx, int worker, double idle_time);
static void pushed_task_cb(unsigned sched_ctx, int worker);
static void poped_task_cb(unsigned sched_ctx, int worker);
static void post_exec_hook_cb(unsigned sched_ctx, int taskid);

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
		break;
	}
}

struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(int type)
{
	hypervisor.nsched_ctxs = 0;
	hypervisor.resize = 0;
	pthread_mutex_init(&act_hypervisor_mutex, NULL);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		hypervisor.configurations[i] = NULL;
		hypervisor.sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].sched_ctx = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].data = NULL;
		int j;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
		{
			hypervisor.sched_ctx_w[i].current_idle_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].tasks[j] = 0;
			hypervisor.sched_ctx_w[i].poped_tasks[j] = 0;
		}
	}

	_load_hypervisor_policy(type);

	hypervisor.policy.init();
	criteria = (struct starpu_sched_ctx_hypervisor_criteria*)malloc(sizeof(struct starpu_sched_ctx_hypervisor_criteria));
	criteria->idle_time_cb = idle_time_cb;
	criteria->pushed_task_cb = pushed_task_cb;
	criteria->poped_task_cb = poped_task_cb;
	criteria->post_exec_hook_cb = post_exec_hook_cb;
	return criteria;
}

void sched_ctx_hypervisor_shutdown(void)
{
	hypervisor.resize = 0;
	hypervisor.policy.deinit();
	free(criteria);
	pthread_mutex_destroy(&act_hypervisor_mutex);
}

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx)
{	
	hypervisor.configurations[sched_ctx] = (struct starpu_htbl32_node_s*)malloc(sizeof(struct starpu_htbl32_node_s));
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
}

void sched_ctx_hypervisor_set_data(unsigned sched_ctx, void *data)
{
	pthread_mutex_lock(&act_hypervisor_mutex);

	if(hypervisor.sched_ctx_w[sched_ctx].data != NULL)
		free(hypervisor.sched_ctx_w[sched_ctx].data);

	hypervisor.sched_ctx_w[sched_ctx].data = data;
	pthread_mutex_unlock(&act_hypervisor_mutex);
	return;
}

void* sched_ctx_hypervisor_get_data(unsigned sched_ctx)
{
	return hypervisor.sched_ctx_w[sched_ctx].data;
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

	/* hypervisor configuration to be considered later */
	void *data = hypervisor.policy.ioctl(sched_ctx, varg_list, (task_tag > 0));
	if(data != NULL)
		_starpu_htbl_insert_32(&hypervisor.configurations[sched_ctx], (uint32_t)task_tag, data);

	return;
}

static void _sched_ctx_hypervisor_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move)
{
	int i;
	printf("resize ctx %d with", sender_sched_ctx);
	for(i = 0; i < nworkers_to_move; i++)
		printf(" %d", workers_to_move[i]);
	printf("\n");

	starpu_remove_workers_from_sched_ctx(workers_to_move, nworkers_to_move, sender_sched_ctx);
	starpu_add_workers_to_sched_ctx(workers_to_move, nworkers_to_move, receiver_sched_ctx);

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

static unsigned check_tasks_of_sched_ctx(unsigned sched_ctx)
{
	int ntasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].tasks);
	
	return ntasks > hypervisor.min_tasks;
}

void sched_ctx_hypervisor_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move)
{
	if(hypervisor.resize)
		_sched_ctx_hypervisor_resize(sender_sched_ctx, receiver_sched_ctx, workers_to_move, nworkers_to_move);

	int i;
	for(i = 0; i < nworkers_to_move; i++)
		hypervisor.sched_ctx_w[sender_sched_ctx].current_idle_time[workers_to_move[i]] = 0.0;

	return;
}

void sched_ctx_hypervisor_advise(unsigned sched_ctx, int *workerids, int nworkers, struct sched_ctx_hypervisor_reply *reply)
{
	pthread_mutex_lock(&act_hypervisor_mutex);
	
	if(hypervisor.sched_ctx_w[sched_ctx].sched_ctx != STARPU_NMAX_SCHED_CTXS)
	{
		starpu_add_workers_to_sched_ctx(workerids, nworkers, sched_ctx);
		
		struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx);
		
		int i = 0;
		
		if(workers->init_cursor)
			workers->init_cursor(workers);
		
		while(workers->has_next(workers))
			reply->procs[i++] = workers->get_next(workers);
		
		if(workers->init_cursor)
			workers->deinit_cursor(workers);
		
		reply->nprocs = i;
		sched_ctx_hypervisor_ioctl(sched_ctx, 
					   HYPERVISOR_PRIORITY, workerids, nworkers, 1,
					   NULL);
		
	}

	pthread_mutex_unlock(&act_hypervisor_mutex);
	return;
}

struct sched_ctx_hypervisor_reply* sched_ctx_hypervisor_request(unsigned sched_ctx, int *workers, int nworkers)
{
	if(hypervisor.sched_ctx_w[sched_ctx].sched_ctx != STARPU_NMAX_SCHED_CTXS)
	{
	}
	return NULL;
}

static void idle_time_cb(unsigned sched_ctx, int worker, double idle_time)
{
	hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker] += idle_time;

	if(hypervisor.nsched_ctxs > 1 && hypervisor.policy.manage_idle_time)
		hypervisor.policy.manage_idle_time(sched_ctx, hypervisor.sched_ctxs, hypervisor.nsched_ctxs, worker, hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker]);
		

	return;
}

static void working_time_cb(unsigned sched_ctx, int worker, double working_time, unsigned current_nprocs)
{
	return;
}


static void pushed_task_cb(unsigned sched_ctx, int worker)
{	
	hypervisor.sched_ctx_w[sched_ctx].tasks[worker]++;
       
	int ntasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].tasks);
	
	hypervisor.resize = (ntasks > hypervisor.min_tasks);
}

static void poped_task_cb(unsigned sched_ctx, int worker)
{
	hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker]++;
	int npoped_tasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].poped_tasks);
       	int ntasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].tasks);
	hypervisor.resize = ((ntasks - npoped_tasks) > 0);
}

static void post_exec_hook_cb(unsigned sched_ctx, int task_tag)
{
	STARPU_ASSERT(task_tag > 0);
	void *data = _starpu_htbl_search_32(hypervisor.configurations[sched_ctx], (uint32_t)task_tag);
	if(data != NULL)	
		sched_ctx_hypervisor_set_data(sched_ctx, data);
}
