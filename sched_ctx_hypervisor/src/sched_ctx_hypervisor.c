#include <sched_ctx_hypervisor_intern.h>

struct starpu_sched_ctx_hypervisor_criteria* criteria = NULL;

extern struct hypervisor_policy simple_policy;

static void idle_time_cb(unsigned sched_ctx, int worker, double idle_time);

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
	hypervisor.resize = 1;
	hypervisor.nsched_ctxs = 0;

	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		hypervisor.sched_ctxs[i] = -1;
		hypervisor.sched_ctx_w[i].sched_ctx = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].data = NULL;
		int j;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
			hypervisor.sched_ctx_w[i].current_idle_time[j] = 0.0;
	}


	_load_hypervisor_policy(type);

	hypervisor.policy.init();
	criteria = (struct starpu_sched_ctx_hypervisor_criteria*)malloc(sizeof(struct starpu_sched_ctx_hypervisor_criteria));
	criteria->idle_time_cb = idle_time_cb;
	return criteria;
}

void sched_ctx_hypervisor_shutdown(void)
{
	hypervisor.policy.deinit();
	hypervisor.resize = 0;
	free(criteria);
}

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx)
{	
	hypervisor.policy.add_sched_ctx(sched_ctx);
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = sched_ctx;
	hypervisor.sched_ctxs[hypervisor.nsched_ctxs++] = sched_ctx;
}

static int _get_first_free_sched_ctx(int *sched_ctxs, unsigned nsched_ctxs)
{
        int i;
        for(i = 0; i < nsched_ctxs; i++)
                if(sched_ctxs[i] == -1)
                        return i;

        return -1;
}

/* rearange array of sched_ctxs in order not to have {-1, -1, 5, -1, 7}    
   and have instead {5, 7, -1, -1, -1}                                    
   it is easier afterwards to iterate the array                           
*/
static void _rearange_sched_ctxs(int *sched_ctxs, int old_nsched_ctxs)
{
        int first_free_id = -1;
        int i;
        for(i = 0; i < old_nsched_ctxs; i++)
        {
                if(sched_ctxs[i] != -1)
                {
                        first_free_id = _get_first_free_sched_ctx(sched_ctxs, old_nsched_ctxs);
                        if(first_free_id != -1)
			{
                                sched_ctxs[first_free_id] = sched_ctxs[i];
				sched_ctxs[i] = -1;
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
                        hypervisor.sched_ctxs[i] = -1;
			break;
                }
        }

        _rearange_sched_ctxs(hypervisor.sched_ctxs, hypervisor.nsched_ctxs);
	hypervisor.nsched_ctxs--;
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = STARPU_NMAX_SCHED_CTXS;
	hypervisor.policy.remove_sched_ctx(sched_ctx);
}

void sched_ctx_hypervisor_set_data(unsigned sched_ctx, void *data)
{
	hypervisor.sched_ctx_w[sched_ctx].data = data;
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

	hypervisor.policy.ioctl(sched_ctx, varg_list);
	return;
}

static void _sched_ctx_hypervisor_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move)
{	
	starpu_remove_workers_from_sched_ctx(workers_to_move, nworkers_to_move, sender_sched_ctx);
	starpu_add_workers_to_sched_ctx(workers_to_move, nworkers_to_move, receiver_sched_ctx);
	
	int i;
	for(i = 0; i < nworkers_to_move; i++)
		hypervisor.sched_ctx_w[sender_sched_ctx].current_idle_time[workers_to_move[i]] = 0.0;

	return;
}

void sched_ctx_hypervisor_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move)
{
	if(hypervisor.resize)
		_sched_ctx_hypervisor_resize(sender_sched_ctx, receiver_sched_ctx, workers_to_move, nworkers_to_move);

	return;
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

