/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <sc_hypervisor_intern.h>

static struct sc_hypervisor_policy_config* _create_config(void)
{
	struct sc_hypervisor_policy_config *config = (struct sc_hypervisor_policy_config *)malloc(sizeof(struct sc_hypervisor_policy_config));
	config->min_nworkers = -1;
	config->max_nworkers = -1;
	config->new_workers_max_idle = -1.0;
	config->ispeed_ctx_sample = 0.0;
	config->time_sample = 0.5;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		config->granularity = -1;
		config->priority[i] = -1;
		config->fixed_workers[i] = -1;
		config->max_idle[i] = -1.0;
		config->min_working[i] = -1.0;
		config->ispeed_w_sample[i] = 0.0;
	}

	return config;
}

static void _update_config(struct sc_hypervisor_policy_config *old, struct sc_hypervisor_policy_config* new)
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
		old->min_working[i] = new->min_working[i] != -1.0 ? new->min_working[i] : old->min_working[i];
	}
}

void sc_hypervisor_set_config(unsigned sched_ctx, void *config)
{
	if(hypervisor.sched_ctx_w[sched_ctx].config != NULL && config != NULL)
	{
		_update_config(hypervisor.sched_ctx_w[sched_ctx].config, config);
	}
	else
	{
		hypervisor.sched_ctx_w[sched_ctx].config = config;
	}

	return;
}

void _add_config(unsigned sched_ctx)
{
	struct sc_hypervisor_policy_config *config = _create_config();
	config->min_nworkers = 0;
	config->max_nworkers = starpu_worker_get_count();
	config->new_workers_max_idle = MAX_IDLE_TIME;

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		config->granularity = 1;
		config->priority[i] = 0;
		config->fixed_workers[i] = 0;
		config->max_idle[i] = MAX_IDLE_TIME;
		config->min_working[i] = MIN_WORKING_TIME;
	}

	sc_hypervisor_set_config(sched_ctx, config);
}

void _remove_config(unsigned sched_ctx)
{
	sc_hypervisor_set_config(sched_ctx, NULL);
}

struct sc_hypervisor_policy_config* sc_hypervisor_get_config(unsigned sched_ctx)
{
	return hypervisor.sched_ctx_w[sched_ctx].config;
}

static struct sc_hypervisor_policy_config* _ctl(unsigned sched_ctx, va_list varg_list, unsigned later)
{
	struct sc_hypervisor_policy_config *config = NULL;

	if(later)
		config = _create_config();
	else
		config = sc_hypervisor_get_config(sched_ctx);

	assert(config != NULL);

	int arg_type;
	int i;
	int *workerids;
	int nworkers;

	while ((arg_type = va_arg(varg_list, int)) != SC_HYPERVISOR_NULL)
	{
		switch(arg_type)
		{
		case SC_HYPERVISOR_MAX_IDLE:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			double max_idle = va_arg(varg_list, double);
			for(i = 0; i < nworkers; i++)
				config->max_idle[workerids[i]] = max_idle;

			break;

		case SC_HYPERVISOR_MIN_WORKING:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			double min_working = va_arg(varg_list, double);

			for(i = 0; i < nworkers; i++)
				config->min_working[workerids[i]] = min_working;

			break;

		case SC_HYPERVISOR_PRIORITY:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			int priority = va_arg(varg_list, int);

			for(i = 0; i < nworkers; i++)
				config->priority[workerids[i]] = priority;
			break;

		case SC_HYPERVISOR_MIN_WORKERS:
			config->min_nworkers = va_arg(varg_list, unsigned);
			break;

		case SC_HYPERVISOR_MAX_WORKERS:
			config->max_nworkers = va_arg(varg_list, unsigned);
			break;

		case SC_HYPERVISOR_GRANULARITY:
			config->granularity = va_arg(varg_list, unsigned);
			break;

		case SC_HYPERVISOR_FIXED_WORKERS:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);

			for(i = 0; i < nworkers; i++)
				config->fixed_workers[workerids[i]] = 1;
			break;

		case SC_HYPERVISOR_NEW_WORKERS_MAX_IDLE:
			config->new_workers_max_idle = va_arg(varg_list, double);
			break;

		case SC_HYPERVISOR_ISPEED_W_SAMPLE:
			workerids = va_arg(varg_list, int*);
			nworkers = va_arg(varg_list, int);
			double sample = va_arg(varg_list, double);

			for(i = 0; i < nworkers; i++)
				config->ispeed_w_sample[workerids[i]] = sample;
			break;

		case SC_HYPERVISOR_ISPEED_CTX_SAMPLE:
			config->ispeed_ctx_sample = va_arg(varg_list, double);
			break;

		case SC_HYPERVISOR_TIME_SAMPLE:
			config->time_sample = va_arg(varg_list, double);
			break;


/* not important for the strateg, needed just to jump these args in the iteration of the args */
		case SC_HYPERVISOR_TIME_TO_APPLY:
			va_arg(varg_list, int);
			break;

		case SC_HYPERVISOR_MIN_TASKS:
			va_arg(varg_list, int);
			break;

		}
	}

	return later ? config : NULL;
}


void sc_hypervisor_ctl(unsigned sched_ctx, ...)
{
	va_list varg_list;
	va_start(varg_list, sched_ctx);

	int arg_type;
	int stop = 0;
	int task_tag = -1;

	while ((arg_type = va_arg(varg_list, int)) != SC_HYPERVISOR_NULL)
	{
		switch(arg_type)
		{
		case SC_HYPERVISOR_TIME_TO_APPLY:
			task_tag = va_arg(varg_list, int);
			stop = 1;
			break;

		case SC_HYPERVISOR_MIN_TASKS:
			hypervisor.min_tasks = va_arg(varg_list, int);
			hypervisor.check_min_tasks[sched_ctx] = 1;
			break;

		}
		if(stop) break;
	}

	va_end(varg_list);
	va_start(varg_list, sched_ctx);

	/* if config not null => save hypervisor configuration and consider it later */
	struct sc_hypervisor_policy_config *config = _ctl(sched_ctx, varg_list, (task_tag > 0));
	if(config != NULL)
	{
		struct configuration_entry *entry;

		entry = malloc(sizeof *entry);
		STARPU_ASSERT(entry != NULL);

		entry->task_tag = task_tag;
		entry->configuration = config;

		STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.conf_mut[sched_ctx]);
		HASH_ADD_INT(hypervisor.configurations[sched_ctx], task_tag, entry);
		STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.conf_mut[sched_ctx]);
	}

	va_end(varg_list);
}
