#include <sched_ctx_hypervisor_intern.h>

static struct policy_config* _create_config(void)
{
	struct policy_config *config = (struct policy_config *)malloc(sizeof(struct policy_config));
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

static void _update_config(struct policy_config *old, struct policy_config* new)
{
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

void sched_ctx_hypervisor_set_config(unsigned sched_ctx, void *config)
{
	printf("%d: ", sched_ctx );
	if(hypervisor.sched_ctx_w[sched_ctx].config != NULL && config != NULL)
	{
		_update_config(hypervisor.sched_ctx_w[sched_ctx].config, config);
	}
	else
		hypervisor.sched_ctx_w[sched_ctx].config = config;
	
	return;
}

void _add_config(unsigned sched_ctx)
{
	struct policy_config *config = _create_config();
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

void _remove_config(unsigned sched_ctx)
{
	sched_ctx_hypervisor_set_config(sched_ctx, NULL);
}

struct policy_config* sched_ctx_hypervisor_get_config(unsigned sched_ctx)
{
	return hypervisor.sched_ctx_w[sched_ctx].config;
}

static struct policy_config* _ioctl(unsigned sched_ctx, va_list varg_list, unsigned later)
{
	struct policy_config *config = NULL;

	if(later)
		config = _create_config();
	else
		config = sched_ctx_hypervisor_get_config(sched_ctx);

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

	return later ? config : NULL;
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
	struct policy_config *config = _ioctl(sched_ctx, varg_list, (task_tag > 0));
	if(config != NULL)
		_starpu_htbl_insert_32(&hypervisor.configurations[sched_ctx], (uint32_t)task_tag, config);

	return;
}
