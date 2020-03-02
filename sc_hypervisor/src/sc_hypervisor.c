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
#include <sc_hypervisor_policy.h>
#include <starpu_config.h>

struct sc_hypervisor hypervisor;
starpu_pthread_mutex_t act_hypervisor_mutex;
double hyp_overhead = 0.0;
unsigned imposed_resize = 0;
unsigned type_of_tasks_known = 0;
struct starpu_sched_ctx_performance_counters* perf_counters = NULL;

static void notify_idle_cycle(unsigned sched_ctx, int worker, double idle_time);
static void notify_pushed_task(unsigned sched_ctx, int worker);
static void notify_post_exec_task(struct starpu_task *task, size_t data_size, uint32_t footprint,
				  int hypervisor_tag, double flops);
static void notify_poped_task(unsigned sched_ctx, int  worker);
static void notify_submitted_job(struct starpu_task *task, unsigned footprint, size_t data_size);
static void notify_empty_ctx(unsigned sched_ctx, struct starpu_task *task);
static void notify_delete_context(unsigned sched_ctx);

extern struct sc_hypervisor_policy idle_policy;
extern struct sc_hypervisor_policy app_driven_policy;
extern struct sc_hypervisor_policy gflops_rate_policy;
#ifdef STARPU_HAVE_GLPK_H
extern struct sc_hypervisor_policy feft_lp_policy;
extern struct sc_hypervisor_policy teft_lp_policy;
extern struct sc_hypervisor_policy ispeed_lp_policy;
extern struct sc_hypervisor_policy throughput_lp_policy;
#endif // STARPU_HAVE_GLPK_
extern struct sc_hypervisor_policy ispeed_policy;
extern struct sc_hypervisor_policy hard_coded_policy;
extern struct sc_hypervisor_policy perf_count_policy;


static struct sc_hypervisor_policy *predefined_policies[] =
{
        &idle_policy,
	&app_driven_policy,
#ifdef STARPU_HAVE_GLPK_H
	&feft_lp_policy,
	&teft_lp_policy,
	&ispeed_lp_policy,
	&throughput_lp_policy,
#endif // STARPU_HAVE_GLPK_H
	&gflops_rate_policy,
	&ispeed_policy,
	&hard_coded_policy,
	&perf_count_policy
};

static void _load_hypervisor_policy(struct sc_hypervisor_policy *policy)
{
	STARPU_ASSERT(policy);

	hypervisor.policy.name = policy->name;
	hypervisor.policy.size_ctxs = policy->size_ctxs;
	hypervisor.policy.resize_ctxs = policy->resize_ctxs;
	hypervisor.policy.handle_poped_task = policy->handle_poped_task;
	hypervisor.policy.handle_pushed_task = policy->handle_pushed_task;
	hypervisor.policy.handle_idle_cycle = policy->handle_idle_cycle;
	hypervisor.policy.handle_idle_end = policy->handle_idle_end;
	hypervisor.policy.handle_post_exec_hook = policy->handle_post_exec_hook;
	hypervisor.policy.handle_submitted_job = policy->handle_submitted_job;
	hypervisor.policy.end_ctx = policy->end_ctx;
	hypervisor.policy.start_ctx = policy->start_ctx;
	hypervisor.policy.init_worker = policy->init_worker;
}


static struct sc_hypervisor_policy *_find_hypervisor_policy_from_name(const char *policy_name)
{

	if (!policy_name)
		return NULL;

	unsigned i;
	for (i = 0; i < sizeof(predefined_policies)/sizeof(predefined_policies[0]); i++)
	{
		struct sc_hypervisor_policy *p;
		p = predefined_policies[i];
		if (p->name)
		{
			if (strcmp(policy_name, p->name) == 0) {
				/* we found a policy with the requested name */
				return p;
			}
		}
	}
	fprintf(stderr, "Warning: hypervisor policy \"%s\" was not found, try \"help\" to get a list\n", policy_name);

	/* nothing was found */
	return NULL;
}

static void display_sched_help_message(void)
{
	const char* policy_name = getenv("SC_HYPERVISOR_POLICY");
	if (policy_name && (strcmp(policy_name, "help") == 0))
	{
		fprintf(stderr, "SC_HYPERVISOR_POLICY can be either of\n");
		/* display the description of all predefined policies */
		unsigned i;
		for (i = 0; i < sizeof(predefined_policies)/sizeof(predefined_policies[0]); i++)
		{
			struct sc_hypervisor_policy *p = predefined_policies[i];
			if (p->name)
			{
				fprintf(stderr, "%s\n", p->name);
			}
		}
	}
}

static struct sc_hypervisor_policy *_select_hypervisor_policy(struct sc_hypervisor_policy* hypervisor_policy)
{
	struct sc_hypervisor_policy *selected_policy = NULL;

	if(hypervisor_policy && hypervisor_policy->custom)
		return hypervisor_policy;

	/* we look if the application specified the name of a policy to load */
	const char *policy_name;
	if (hypervisor_policy && hypervisor_policy->name)
	{
		policy_name = hypervisor_policy->name;
	}
	else
	{
		policy_name = getenv("SC_HYPERVISOR_POLICY");
	}

	if (policy_name)
		selected_policy = _find_hypervisor_policy_from_name(policy_name);

	/* Perhaps there was no policy that matched the name */
	if (selected_policy)
		return selected_policy;

	/* If no policy was specified, we use the idle policy as a default */

	return &idle_policy;
}


/* initializez the performance counters that starpu will use to retrive hints for resizing */
void* sc_hypervisor_init(struct sc_hypervisor_policy *hypervisor_policy)
{
/* Perhaps we have to display some help */
	display_sched_help_message();

	hypervisor.min_tasks = 0;
	hypervisor.nsched_ctxs = 0;
	char* vel_gap = getenv("SC_HYPERVISOR_MAX_SPEED_GAP");
	hypervisor.max_speed_gap = vel_gap ? atof(vel_gap) : SC_SPEED_MAX_GAP_DEFAULT;
	char* crit =  getenv("SC_HYPERVISOR_TRIGGER_RESIZE");
	hypervisor.resize_criteria = !crit ? SC_IDLE : strcmp(crit,"idle") == 0 ? SC_IDLE : (strcmp(crit,"speed") == 0 ? SC_SPEED : SC_NOTHING);

	STARPU_PTHREAD_MUTEX_INIT(&act_hypervisor_mutex, NULL);
//	hypervisor.start_executing_time = starpu_timing_now();

	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		hypervisor.resize[i] = 0;
		hypervisor.allow_remove[i] = 1;
		hypervisor.configurations[i] = NULL;
		hypervisor.sr = NULL;
		hypervisor.check_min_tasks[i] = 1;
		hypervisor.sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].sched_ctx = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].config = NULL;
		hypervisor.sched_ctx_w[i].total_flops = 0.0;
		hypervisor.sched_ctx_w[i].submitted_flops = 0.0;
		hypervisor.sched_ctx_w[i].remaining_flops = 0.0;
		hypervisor.sched_ctx_w[i].start_time = 0.0;
		hypervisor.sched_ctx_w[i].real_start_time = 0.0;
		hypervisor.sched_ctx_w[i].hyp_react_start_time = 0.0;
		hypervisor.sched_ctx_w[i].resize_ack.receiver_sched_ctx = -1;
		hypervisor.sched_ctx_w[i].resize_ack.moved_workers = NULL;
		hypervisor.sched_ctx_w[i].resize_ack.nmoved_workers = 0;
		hypervisor.sched_ctx_w[i].resize_ack.acked_workers = NULL;
		STARPU_PTHREAD_MUTEX_INIT(&hypervisor.sched_ctx_w[i].mutex, NULL);
		hypervisor.optimal_v[i] = 0.0;

		hypervisor.sched_ctx_w[i].ref_speed[0] = -1.0;
		hypervisor.sched_ctx_w[i].ref_speed[1] = -1.0;
		hypervisor.sched_ctx_w[i].total_flops_available = 0;
		hypervisor.sched_ctx_w[i].to_be_sized = 0;
		hypervisor.sched_ctx_w[i].consider_max = 0;
		int j;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
		{
			hypervisor.sched_ctx_w[i].start_time_w[i] = 0.0;
			hypervisor.sched_ctx_w[i].current_idle_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].idle_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].idle_start_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].exec_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].exec_start_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].pushed_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].poped_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].elapsed_flops[j] = 0.0;
			hypervisor.sched_ctx_w[i].elapsed_data[j] = 0;
			hypervisor.sched_ctx_w[i].elapsed_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].total_elapsed_flops[j] = 0.0;
			hypervisor.sched_ctx_w[i].worker_to_be_removed[j] = 0;
			hypervisor.sched_ctx_w[i].compute_idle[j] = 1;
			hypervisor.sched_ctx_w[i].compute_partial_idle[j] = 0;
		}
	}

	struct sc_hypervisor_policy *selected_hypervisor_policy = _select_hypervisor_policy(hypervisor_policy);
	_load_hypervisor_policy(selected_hypervisor_policy);

	perf_counters = (struct starpu_sched_ctx_performance_counters*)malloc(sizeof(struct starpu_sched_ctx_performance_counters));
	perf_counters->notify_idle_cycle = notify_idle_cycle;
	perf_counters->notify_pushed_task = notify_pushed_task;
	perf_counters->notify_poped_task = notify_poped_task;
	perf_counters->notify_post_exec_task = notify_post_exec_task;
	perf_counters->notify_submitted_job = notify_submitted_job;
	perf_counters->notify_empty_ctx = notify_empty_ctx;
	perf_counters->notify_delete_context = notify_delete_context;

	starpu_sched_ctx_notify_hypervisor_exists();

	return (void*)perf_counters;
}

const char* sc_hypervisor_get_policy()
{
	return hypervisor.policy.name;
}

/* the user can forbid the resizing process*/
void sc_hypervisor_stop_resize(unsigned sched_ctx)
{
	imposed_resize = 1;
	hypervisor.resize[sched_ctx] = 0;
}

/* the user can restart the resizing process*/
void sc_hypervisor_start_resize(unsigned sched_ctx)
{
	imposed_resize = 1;
	hypervisor.resize[sched_ctx] = 1;
}

static void _print_current_time()
{
	char* stop_print = getenv("SC_HYPERVISOR_STOP_PRINT");
        int sp = stop_print ? atoi(stop_print) : 1;

	if(!sp)
	{
		if(hypervisor.start_executing_time == 0.0)
		{
			fprintf(stdout, "Time: %lf\n", -1.0);
			return;
		}

		double curr_time = starpu_timing_now();
		double elapsed_time = (curr_time - hypervisor.start_executing_time) / 1000000.0; /* in seconds */
		fprintf(stdout, "Time: %lf\n", elapsed_time);
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS)
			{
				struct sc_hypervisor_wrapper *sc_w = &hypervisor.sched_ctx_w[hypervisor.sched_ctxs[i]];

				double cpu_speed = sc_hypervisor_get_speed(sc_w, STARPU_CPU_WORKER);
				double cuda_speed = sc_hypervisor_get_speed(sc_w, STARPU_CUDA_WORKER);
				int ncpus = sc_hypervisor_get_nworkers_ctx(sc_w->sched_ctx, STARPU_CPU_WORKER);
				int ncuda = sc_hypervisor_get_nworkers_ctx(sc_w->sched_ctx, STARPU_CUDA_WORKER);
				fprintf(stdout, "%d: cpu_v = %lf cuda_v = %lf ncpus = %d ncuda = %d\n", hypervisor.sched_ctxs[i], cpu_speed, cuda_speed, ncpus, ncuda);
			}
		}
	}
	return;
}

void sc_hypervisor_shutdown(void)
{
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
                if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && hypervisor.nsched_ctxs > 0)
		{
			sc_hypervisor_stop_resize(hypervisor.sched_ctxs[i]);
			sc_hypervisor_unregister_ctx(hypervisor.sched_ctxs[i]);
			STARPU_PTHREAD_MUTEX_DESTROY(&hypervisor.sched_ctx_w[i].mutex);
		}
	}
	perf_counters->notify_idle_cycle = NULL;
	perf_counters->notify_pushed_task = NULL;
	perf_counters->notify_poped_task = NULL;
	perf_counters->notify_post_exec_task = NULL;
	perf_counters->notify_delete_context = NULL;

	free(perf_counters);
	perf_counters = NULL;

	STARPU_PTHREAD_MUTEX_DESTROY(&act_hypervisor_mutex);

}

void sc_hypervisor_print_overhead()
{
//	hyp_overhead /= 1000000.0;*
	FILE *f;
	const char *sched_env = getenv("OVERHEAD_FILE");
	if(!sched_env)
		f = fopen("overhead_microsec", "a");
	else
		f = fopen(sched_env, "a");
	fprintf(f, "%lf \n", hyp_overhead);
	fclose(f);


}

/* the hypervisor is in charge only of the contexts registered to it*/
void sc_hypervisor_register_ctx(unsigned sched_ctx, double total_flops)
{
	if(hypervisor.policy.start_ctx)
		hypervisor.policy.start_ctx(sched_ctx);

	STARPU_PTHREAD_MUTEX_LOCK(&act_hypervisor_mutex);
	hypervisor.configurations[sched_ctx] = NULL;
	hypervisor.resize_requests[sched_ctx] = NULL;
	STARPU_PTHREAD_MUTEX_INIT(&hypervisor.conf_mut[sched_ctx], NULL);
	STARPU_PTHREAD_MUTEX_INIT(&hypervisor.resize_mut[sched_ctx], NULL);

	_add_config(sched_ctx);
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = sched_ctx;
	hypervisor.sched_ctxs[hypervisor.nsched_ctxs++] = sched_ctx;

	hypervisor.sched_ctx_w[sched_ctx].total_flops = total_flops;
	hypervisor.sched_ctx_w[sched_ctx].remaining_flops = total_flops;
	hypervisor.resize[sched_ctx] = 0;//1;
	STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
}

static int _get_first_free_sched_ctx(unsigned *sched_ctxs, int nsched_ctxs)
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
static void _rearange_sched_ctxs(unsigned *sched_ctxs, int old_nsched_ctxs)
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

/* unregistered contexts will no longer be resized */
void sc_hypervisor_unregister_ctx(unsigned sched_ctx)
{
#ifdef STARPU_SC_HYPERVISOR_DEBUG
	printf("unregister ctx %d with remaining flops %lf \n", hypervisor.sched_ctx_w[sched_ctx].sched_ctx, hypervisor.sched_ctx_w[sched_ctx].remaining_flops);
#endif
	if(hypervisor.policy.end_ctx)
		hypervisor.policy.end_ctx(sched_ctx);

	STARPU_PTHREAD_MUTEX_LOCK(&act_hypervisor_mutex);
	unsigned father = starpu_sched_ctx_get_inheritor(sched_ctx);
	int *pus;
	unsigned npus = starpu_sched_ctx_get_workers_list(sched_ctx, &pus);

	if(npus)
	{
		starpu_sched_ctx_set_priority(pus, npus, father, 1);
		free(pus);
	}

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
	_remove_config(sched_ctx);

	STARPU_PTHREAD_MUTEX_DESTROY(&hypervisor.conf_mut[sched_ctx]);
	STARPU_PTHREAD_MUTEX_DESTROY(&hypervisor.resize_mut[sched_ctx]);
	if(hypervisor.nsched_ctxs == 1)
		sc_hypervisor_stop_resize(hypervisor.sched_ctxs[0]);

	STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
}

void sc_hypervisor_reset_react_start_time(unsigned sched_ctx, unsigned now)
{
	if(now)
		hypervisor.sched_ctx_w[sched_ctx].hyp_react_start_time = starpu_timing_now();
	starpu_sched_ctx_update_start_resizing_sample(sched_ctx, starpu_timing_now());
}


double _get_max_speed_gap()
{
	return hypervisor.max_speed_gap;
}

unsigned sc_hypervisor_get_resize_criteria()
{
	return hypervisor.resize_criteria;
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

int sc_hypervisor_get_nworkers_ctx(unsigned sched_ctx, enum starpu_worker_archtype arch)
{
	int nworkers_ctx = 0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		enum starpu_worker_archtype curr_arch = starpu_worker_get_type(worker);
		if(curr_arch == arch || arch == STARPU_ANY_WORKER)
			nworkers_ctx++;
	}
	return nworkers_ctx;
}

static void _set_elapsed_flops_per_sched_ctx(unsigned sched_ctx, double val)
{
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
	{
		hypervisor.sched_ctx_w[sched_ctx].elapsed_flops[i] = val;
		if(val == 0)
		{
			hypervisor.sched_ctx_w[sched_ctx].elapsed_data[i] = 0;
			hypervisor.sched_ctx_w[sched_ctx].elapsed_tasks[i] = 0;
		}
	}
}

double sc_hypervisor_get_elapsed_flops_per_sched_ctx(struct sc_hypervisor_wrapper* sc_w)
{
	double ret_val = 0.0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		ret_val += sc_w->elapsed_flops[worker];
	}

	return ret_val;
}

double sc_hypervisor_get_total_elapsed_flops_per_sched_ctx(struct sc_hypervisor_wrapper* sc_w)
{
	double ret_val = 0.0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		ret_val += sc_w->total_elapsed_flops[worker];
	}

	return ret_val;
}

double sc_hypervisor_get_nready_flops_of_all_sons_of_sched_ctx(unsigned sched_ctx)
{
	double ready_flops = starpu_sched_ctx_get_nready_flops(sched_ctx);
	unsigned *sched_ctxs;
	int nsched_ctxs = 0;
	sc_hypervisor_get_ctxs_on_level(&sched_ctxs, &nsched_ctxs, starpu_sched_ctx_get_hierarchy_level(sched_ctx), sched_ctx);
	int s;
	for(s = 0; s < nsched_ctxs; s++)
		ready_flops += sc_hypervisor_get_nready_flops_of_all_sons_of_sched_ctx(sched_ctxs[s]);
		//ready_flops += starpu_get_nready_flops_of_sched_ctx(sched_ctxs[s]);

	free(sched_ctxs);
	return ready_flops;
}
static void _decrement_elapsed_flops_per_worker(unsigned sched_ctx, int worker, double flops)
{
	if(starpu_sched_ctx_get_hierarchy_level(sched_ctx) > 0)
	{
		unsigned father = starpu_sched_ctx_get_inheritor(sched_ctx);
		hypervisor.sched_ctx_w[father].elapsed_flops[worker] -= flops;
		_decrement_elapsed_flops_per_worker(father, worker, flops);
	}

	return;
}
void _reset_resize_sample_info(unsigned sender_sched_ctx, unsigned receiver_sched_ctx)
{
	double start_time =  starpu_timing_now();
	if(sender_sched_ctx != STARPU_NMAX_SCHED_CTXS)
	{
		/* info concerning only the gflops_rate strateg */
		struct sc_hypervisor_wrapper *sender_sc_w = &hypervisor.sched_ctx_w[sender_sched_ctx];

		sender_sc_w->start_time = start_time;
		unsigned nworkers = starpu_worker_get_count();
		int i;
 		for(i = 0; i < nworkers; i++)
		{
			sender_sc_w->start_time_w[i] = start_time;
			sender_sc_w->idle_time[i] = 0.0;
			sender_sc_w->idle_start_time[i] = 0.0;
			hypervisor.sched_ctx_w[sender_sched_ctx].exec_time[i] = 0.0;
//			hypervisor.sched_ctx_w[sender_sched_ctx].exec_start_time[i] = (hypervisor.sched_ctx_w[sender_sched_ctx].exec_start_time[i] != 0.0) ? starpu_timing_now() : 0.0;
			_decrement_elapsed_flops_per_worker(sender_sched_ctx, i, hypervisor.sched_ctx_w[sender_sched_ctx].elapsed_flops[i]);

		}
		_set_elapsed_flops_per_sched_ctx(sender_sched_ctx, 0.0);
	}

	if(receiver_sched_ctx != STARPU_NMAX_SCHED_CTXS)
	{
		struct sc_hypervisor_wrapper *receiver_sc_w = &hypervisor.sched_ctx_w[receiver_sched_ctx];

		receiver_sc_w->start_time = start_time;

		unsigned nworkers = starpu_worker_get_count();
		int i;
 		for(i = 0; i < nworkers; i++)
		{
			receiver_sc_w->start_time_w[i] = (receiver_sc_w->start_time_w[i] != 0.0) ? starpu_timing_now() : 0.0;
			receiver_sc_w->idle_time[i] = 0.0;
			receiver_sc_w->idle_start_time[i] = (receiver_sc_w->exec_start_time[i] != 0.0) ? 0.0 : starpu_timing_now();
//			hypervisor.sched_ctx_w[receiver_sched_ctx].exec_start_time[i] = (receiver_sc_w->exec_start_time[i] != 0.0) ? starpu_timing_now() : 0.0;
			hypervisor.sched_ctx_w[receiver_sched_ctx].exec_time[i] = 0.0;
			_decrement_elapsed_flops_per_worker(receiver_sched_ctx, i, hypervisor.sched_ctx_w[receiver_sched_ctx].elapsed_flops[i]);
		}
		_set_elapsed_flops_per_sched_ctx(receiver_sched_ctx, 0.0);
	}
	return;
}

/* actually move the workers: the cpus are moved, gpus are only shared  */
/* forbids another resize request before this one is take into account */
void sc_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move, unsigned now)
{
	if(nworkers_to_move > 0 && hypervisor.resize[sender_sched_ctx])
	{
		_print_current_time();
		unsigned j;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
		printf("resize ctx %u with %u workers", sender_sched_ctx, nworkers_to_move);
		for(j = 0; j < nworkers_to_move; j++)
			printf(" %d", workers_to_move[j]);
		printf("\n");
#endif

		hypervisor.allow_remove[receiver_sched_ctx] = 0;
		starpu_sched_ctx_add_workers(workers_to_move, nworkers_to_move, receiver_sched_ctx);

		if(now)
		{
			unsigned j;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
			printf("remove now from ctx %u:", sender_sched_ctx);
			for(j = 0; j < nworkers_to_move; j++)
				printf(" %d", workers_to_move[j]);
			printf("\n");
#endif
			starpu_sched_ctx_remove_workers(workers_to_move, nworkers_to_move, sender_sched_ctx);
			hypervisor.allow_remove[receiver_sched_ctx] = 1;
			_reset_resize_sample_info(sender_sched_ctx, receiver_sched_ctx);
		}
		else
		{
			int ret = starpu_pthread_mutex_trylock(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
			if(ret != EBUSY)
			{
				hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.receiver_sched_ctx = receiver_sched_ctx;
				hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.moved_workers = (int*)malloc(nworkers_to_move * sizeof(int));
				hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.nmoved_workers = nworkers_to_move;
				hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.acked_workers = (int*)malloc(nworkers_to_move * sizeof(int));


				unsigned i;
				for(i = 0; i < nworkers_to_move; i++)
				{
					hypervisor.sched_ctx_w[sender_sched_ctx].current_idle_time[workers_to_move[i]] = 0.0;
					hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.moved_workers[i] = workers_to_move[i];
					hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.acked_workers[i] = 0;
				}

				hypervisor.resize[sender_sched_ctx] = 0;
				if(imposed_resize)  imposed_resize = 0;
				STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
			}
		}
		struct sc_hypervisor_policy_config *new_config = sc_hypervisor_get_config(receiver_sched_ctx);
		unsigned i;
		for(i = 0; i < nworkers_to_move; i++)
			new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;

	}
	return;
}

void sc_hypervisor_add_workers_to_sched_ctx(int* workers_to_add, unsigned nworkers_to_add, unsigned sched_ctx)
{
	if(nworkers_to_add > 0 && hypervisor.resize[sched_ctx])
	{
		_print_current_time();
		unsigned j;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
		printf("add to ctx %u:", sched_ctx);
		for(j = 0; j < nworkers_to_add; j++)
			printf(" %d", workers_to_add[j]);
		printf("\n");
#endif
		starpu_sched_ctx_add_workers(workers_to_add, nworkers_to_add, sched_ctx);
		struct sc_hypervisor_policy_config *new_config = sc_hypervisor_get_config(sched_ctx);
		unsigned i;
		for(i = 0; i < nworkers_to_add; i++)
			new_config->max_idle[workers_to_add[i]] = new_config->max_idle[workers_to_add[i]] != MAX_IDLE_TIME ? new_config->max_idle[workers_to_add[i]] :  new_config->new_workers_max_idle;
		_reset_resize_sample_info(STARPU_NMAX_SCHED_CTXS, sched_ctx);

	}
	return;
}

unsigned sc_hypervisor_can_resize(unsigned sched_ctx)
{
	return hypervisor.resize[sched_ctx];
}

void sc_hypervisor_remove_workers_from_sched_ctx(int* workers_to_remove, unsigned nworkers_to_remove, unsigned sched_ctx, unsigned now)
{
	if(nworkers_to_remove > 0 && hypervisor.resize[sched_ctx] && hypervisor.allow_remove[sched_ctx])
	{
		_print_current_time();
		unsigned nworkers = 0;
		int workers[nworkers_to_remove];

		if(now)
		{
			unsigned j;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
			printf("remove explicitley now from ctx %u:", sched_ctx);
			for(j = 0; j < nworkers_to_remove; j++)
				printf(" %d", workers_to_remove[j]);
			printf("\n");
#endif
			starpu_sched_ctx_remove_workers(workers_to_remove, nworkers_to_remove, sched_ctx);
			_reset_resize_sample_info(sched_ctx, STARPU_NMAX_SCHED_CTXS);
		}
		else
		{
#ifdef STARPU_SC_HYPERVISOR_DEBUG
			printf("try to remove from ctx %u: ", sched_ctx);
			unsigned j;
			for(j = 0; j < nworkers_to_remove; j++)
				printf(" %d", workers_to_remove[j]);
			printf("\n");
#endif
			int ret = starpu_pthread_mutex_trylock(&hypervisor.sched_ctx_w[sched_ctx].mutex);
			if(ret != EBUSY)
			{

				unsigned i;
				for(i = 0; i < nworkers_to_remove; i++)
					if(starpu_sched_ctx_contains_worker(workers_to_remove[i], sched_ctx))
						workers[nworkers++] = workers_to_remove[i];

				hypervisor.sched_ctx_w[sched_ctx].resize_ack.receiver_sched_ctx = -1;
				hypervisor.sched_ctx_w[sched_ctx].resize_ack.moved_workers = (int*)malloc(nworkers_to_remove * sizeof(int));
				hypervisor.sched_ctx_w[sched_ctx].resize_ack.nmoved_workers = (int)nworkers;
				hypervisor.sched_ctx_w[sched_ctx].resize_ack.acked_workers = (int*)malloc(nworkers_to_remove * sizeof(int));


				for(i = 0; i < nworkers; i++)
				{
					hypervisor.sched_ctx_w[sched_ctx].current_idle_time[workers[i]] = 0.0;
					hypervisor.sched_ctx_w[sched_ctx].resize_ack.moved_workers[i] = workers[i];
					hypervisor.sched_ctx_w[sched_ctx].resize_ack.acked_workers[i] = 0;
				}

				hypervisor.resize[sched_ctx] = 0;
				if(imposed_resize)  imposed_resize = 0;
				STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
			}
		}
 	}
	return;
}

static unsigned _ack_resize_completed(unsigned sched_ctx, int worker)
{
	if(worker != -1 && !starpu_sched_ctx_contains_worker(worker, sched_ctx))
		return 0;

	struct sc_hypervisor_resize_ack *resize_ack = NULL;
	unsigned sender_sched_ctx = STARPU_NMAX_SCHED_CTXS;

	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS)
		{
			struct sc_hypervisor_wrapper *sc_w = &hypervisor.sched_ctx_w[hypervisor.sched_ctxs[i]];
			STARPU_PTHREAD_MUTEX_LOCK(&sc_w->mutex);
			unsigned only_remove = 0;
			if(sc_w->resize_ack.receiver_sched_ctx == -1 && hypervisor.sched_ctxs[i] != sched_ctx &&
			   sc_w->resize_ack.nmoved_workers > 0 && starpu_sched_ctx_contains_worker(worker, hypervisor.sched_ctxs[i]))
			{
				int j;
				for(j = 0; j < sc_w->resize_ack.nmoved_workers; j++)
					if(sc_w->resize_ack.moved_workers[j] == worker)
					{
						only_remove = 1;
						_reset_resize_sample_info(sched_ctx, STARPU_NMAX_SCHED_CTXS);
						break;
					}
			}
			if(only_remove ||
			   (sc_w->resize_ack.receiver_sched_ctx != -1 && sc_w->resize_ack.receiver_sched_ctx == (int)sched_ctx))
			{
				resize_ack = &sc_w->resize_ack;
				sender_sched_ctx = hypervisor.sched_ctxs[i];
				STARPU_PTHREAD_MUTEX_UNLOCK(&sc_w->mutex);
				break;
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&sc_w->mutex);
		}
	}

	/* if there is no ctx waiting for its ack return 1*/
	if(resize_ack == NULL)
	{
		return 1;
	}

	int ret = starpu_pthread_mutex_trylock(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
	if(ret != EBUSY)
	{
		int *moved_workers = resize_ack->moved_workers;
		int nmoved_workers = resize_ack->nmoved_workers;
		int *acked_workers = resize_ack->acked_workers;

		if(worker != -1)
		{
			for(i = 0; i < nmoved_workers; i++)
			{
				int moved_worker = moved_workers[i];
				if(moved_worker == worker && acked_workers[i] == 0)
				{
					acked_workers[i] = 1;
				}
			}
		}

		int nacked_workers = 0;
		for(i = 0; i < nmoved_workers; i++)
		{
			nacked_workers += (acked_workers[i] == 1);
		}

		unsigned resize_completed = (nacked_workers == nmoved_workers);
		int receiver_sched_ctx = sched_ctx;
		if(resize_completed)
		{
			/* if the permission to resize is not allowed by the user don't do it
			   whatever the application says */
			if(!((hypervisor.resize[sender_sched_ctx] == 0 || hypervisor.resize[receiver_sched_ctx] == 0) && imposed_resize))
			{
/* 				int j; */
/* 				printf("remove after ack from ctx %d:", sender_sched_ctx); */
/* 				for(j = 0; j < nmoved_workers; j++) */
/* 					printf(" %d", moved_workers[j]); */
/* 				printf("\n"); */

				starpu_sched_ctx_remove_workers(moved_workers, nmoved_workers, sender_sched_ctx);

				_reset_resize_sample_info(sender_sched_ctx, receiver_sched_ctx);

				hypervisor.resize[sender_sched_ctx] = 1;
				hypervisor.allow_remove[receiver_sched_ctx] = 1;
				/* if the user allowed resizing leave the decisions to the application */
				if(imposed_resize)  imposed_resize = 0;

				resize_ack->receiver_sched_ctx = -1;
				resize_ack->nmoved_workers = 0;
				free(resize_ack->moved_workers);
				free(resize_ack->acked_workers);

			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
			return resize_completed;
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
	}
	return 0;
}

/* Enqueue a resize request for 'sched_ctx', to be executed when the
 * 'task_tag' tasks of 'sched_ctx' complete.  */
void sc_hypervisor_post_resize_request(unsigned sched_ctx, int task_tag)
{
	struct resize_request_entry *entry;

	entry = malloc(sizeof *entry);
	STARPU_ASSERT(entry != NULL);

	entry->sched_ctx = sched_ctx;
	entry->task_tag = task_tag;

	STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.resize_mut[sched_ctx]);
	HASH_ADD_INT(hypervisor.resize_requests[sched_ctx], task_tag, entry);
	STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.resize_mut[sched_ctx]);
}

void sc_hypervisor_resize_ctxs(unsigned *sched_ctxs, int nsched_ctxs , int *workers, int nworkers)
{
	if(hypervisor.policy.resize_ctxs)
		hypervisor.policy.resize_ctxs(sched_ctxs, nsched_ctxs, workers, nworkers);
}

void _sc_hypervisor_allow_compute_idle(unsigned sched_ctx, int worker, unsigned allow)
{
	hypervisor.sched_ctx_w[sched_ctx].compute_idle[worker] = allow;
}


int _update_max_hierarchically(unsigned *sched_ctxs, int nsched_ctxs)
{
	int s;
	unsigned leaves[hypervisor.nsched_ctxs];
	int nleaves = 0;
	sc_hypervisor_get_leaves(hypervisor.sched_ctxs, hypervisor.nsched_ctxs, leaves, &nleaves);

	int max = 0;

	for(s = 0; s < nsched_ctxs; s++)
	{
		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctxs[s]);
		unsigned found = 0;
		int l = 0;
		for(l = 0; l < nleaves; l++)
		{
			if(leaves[l] == sched_ctxs[s])
			{
				found = 1;
				break;
			}
		}
		if(!found)
		{
			config->max_nworkers = 0;
			int level = starpu_sched_ctx_get_hierarchy_level(sched_ctxs[s]);
			unsigned *sched_ctxs_child;
			int nsched_ctxs_child = 0;
			sc_hypervisor_get_ctxs_on_level(&sched_ctxs_child, &nsched_ctxs_child, level+1, sched_ctxs[s]);
			if(nsched_ctxs_child > 0)
			{
				config->max_nworkers += _update_max_hierarchically(sched_ctxs_child, nsched_ctxs_child);
				free(sched_ctxs_child);
				int max_possible_workers = starpu_worker_get_count();
				if(config->max_nworkers < 0)
					config->max_nworkers = 0;
				if(config->max_nworkers > max_possible_workers)
					config->max_nworkers = max_possible_workers;

			}
#ifdef STARPU_SC_HYPERVISOR_DEBUG
			printf("ctx %u has max %d \n", sched_ctxs[s], config->max_nworkers);
#endif
		}
		max += config->max_nworkers;
	}
	return max;
}
void _update_max_diff_hierarchically(unsigned father, double diff)
{
	int level = starpu_sched_ctx_get_hierarchy_level(father);
	unsigned *sched_ctxs_child;
	int nsched_ctxs_child = 0;
	sc_hypervisor_get_ctxs_on_level(&sched_ctxs_child, &nsched_ctxs_child, level+1, father);
	if(nsched_ctxs_child > 0)
	{
		int s;
		double total_nflops = 0.0;
		for(s = 0; s < nsched_ctxs_child; s++)
		{
			total_nflops += hypervisor.sched_ctx_w[sched_ctxs_child[s]].remaining_flops < 0.0 ? 0.0 : hypervisor.sched_ctx_w[sched_ctxs_child[s]].remaining_flops;
		}

		int accumulated_diff = 0;
		for(s = 0; s < nsched_ctxs_child; s++)
		{
			struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctxs_child[s]);
			double remaining_flops = hypervisor.sched_ctx_w[sched_ctxs_child[s]].remaining_flops < 0.0 ? 0.0 : hypervisor.sched_ctx_w[sched_ctxs_child[s]].remaining_flops;
 			int current_diff = total_nflops == 0.0 ? 0.0 : floor((remaining_flops / total_nflops) * diff);
			accumulated_diff += current_diff;
			if(s == (nsched_ctxs_child - 1) && accumulated_diff < diff)
				current_diff += (diff - accumulated_diff);
			config->max_nworkers += current_diff;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
			printf("%u: redib max_nworkers incr %d diff = %d \n",  sched_ctxs_child[s], config->max_nworkers, current_diff);
#endif
			_update_max_diff_hierarchically(sched_ctxs_child[s], current_diff);
		}
		free(sched_ctxs_child);
	}
	return;
}

void sc_hypervisor_update_resize_interval(unsigned *sched_ctxs, int nsched_ctxs, int max_workers)
{
	unsigned leaves[hypervisor.nsched_ctxs];
	unsigned nleaves = 0;
	sc_hypervisor_get_leaves(hypervisor.sched_ctxs, hypervisor.nsched_ctxs, leaves, &nleaves);
	int l;

	unsigned sched_ctx;
	int total_max_nworkers = 0;
//	int max_cpus = starpu_cpu_worker_get_count();
	unsigned configured = 0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		unsigned found = 0;
		for(l = 0; l < nleaves; l++)
		{
			if(leaves[l] == sched_ctxs[i])
			{
				found = 1;
				break;
			}
		}
		if(!found)
			continue;

		sched_ctx = sched_ctxs[i];

		if(hypervisor.sched_ctx_w[sched_ctx].to_be_sized) continue;

		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctx);
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
		int worker;

		struct starpu_sched_ctx_iterator it;
		workers->init_iterator(workers, &it);

		double elapsed_time_worker[STARPU_NMAXWORKERS];
		double norm_idle_time = 0.0;
		double end_time  = starpu_timing_now();
		while(workers->has_next(workers, &it))
		{
			double idle_time = 0.0;
			worker = workers->get_next(workers, &it);
			if(hypervisor.sched_ctx_w[sched_ctx].compute_idle[worker])
			{
				if(hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker] == 0.0)
					elapsed_time_worker[worker] = 0.0;
				else
					elapsed_time_worker[worker] = (end_time - hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker]) / 1000000.0;

				if(hypervisor.sched_ctx_w[sched_ctx].idle_start_time[worker] == 0.0)
				{
					idle_time = hypervisor.sched_ctx_w[sched_ctx].idle_time[worker]; /* in seconds */
				}
				else
				{
					double idle = (end_time - hypervisor.sched_ctx_w[sched_ctx].idle_start_time[worker]) / 1000000.0; /* in seconds */
					idle_time = hypervisor.sched_ctx_w[sched_ctx].idle_time[worker] + idle;
				}
				norm_idle_time += (elapsed_time_worker[worker] == 0.0 ? 0.0 : (idle_time / elapsed_time_worker[worker]));
/* 				printf("%d/%d: start time %lf elapsed time %lf idle time %lf norm_idle_time %lf \n",  */
/* 				       worker, sched_ctx, hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker], elapsed_time_worker[worker], idle_time, norm_idle_time); */
			}
		}

		double norm_exec_time = 0.0;
		for(worker = 0; worker < STARPU_NMAXWORKERS; worker++)
		{
			double exec_time = 0.0;
			if(hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker] == 0.0)
				elapsed_time_worker[worker] = 0.0;
			else
				elapsed_time_worker[worker] = (end_time - hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker]) / 1000000.0;

			if(hypervisor.sched_ctx_w[sched_ctx].exec_start_time[worker] == 0.0)
			{
				exec_time = hypervisor.sched_ctx_w[sched_ctx].exec_time[worker];
			}
			else
			{
				double current_exec_time = 0.0;
				if(hypervisor.sched_ctx_w[sched_ctx].exec_start_time[worker] < hypervisor.sched_ctx_w[sched_ctx].start_time)
					current_exec_time = (end_time - hypervisor.sched_ctx_w[sched_ctx].start_time) / 1000000.0; /* in seconds */
				else
					current_exec_time = (end_time - hypervisor.sched_ctx_w[sched_ctx].exec_start_time[worker]) / 1000000.0; /* in seconds */

				exec_time = hypervisor.sched_ctx_w[sched_ctx].exec_time[worker] + current_exec_time;
			}
			norm_exec_time += elapsed_time_worker[worker] == 0.0 ? 0.0 : exec_time / elapsed_time_worker[worker];
		}

		double curr_time = starpu_timing_now();
		double elapsed_time = (curr_time - hypervisor.sched_ctx_w[sched_ctx].start_time) / 1000000.0; /* in seconds */
		int nready_tasks = starpu_sched_ctx_get_nready_tasks(sched_ctx);
/* 		if(norm_idle_time >= 0.9) */
/* 		{ */
/* 			config->max_nworkers = lrint(norm_exec_time); */
/* 		} */
/* 		else */
/* 		{ */
/* 			if(norm_idle_time < 0.1) */
/* 				config->max_nworkers = lrint(norm_exec_time)  + nready_tasks - 1; //workers->nworkers + hypervisor.sched_ctx_w[sched_ctx].nready_tasks - 1; */
/* 			else */
/* 				config->max_nworkers = lrint(norm_exec_time); */
/* 		} */
		config->max_nworkers = lrint(norm_exec_time);
//		config->max_nworkers = hypervisor.sched_ctx_w[sched_ctx].nready_tasks - 1;

		/* if(config->max_nworkers < 0) */
/* 			config->max_nworkers = 0; */
/* 		if(config->max_nworkers > max_workers) */
/* 			config->max_nworkers = max_workers; */

#ifdef STARPU_SC_HYPERVISOR_DEBUG
		printf("%u: ready tasks  %d norm_idle_time %lf elapsed_time %lf norm_exec_time %lf nworker %d max %d \n",
		       sched_ctx, nready_tasks, norm_idle_time, elapsed_time, norm_exec_time, workers->nworkers, config->max_nworkers);
#endif

		total_max_nworkers += config->max_nworkers;
		configured = 1;

	}

	unsigned nhierarchy_levels = sc_hypervisor_get_nhierarchy_levels();
	if(nhierarchy_levels > 1 && configured)
	{
		unsigned *sched_ctxs2;
		int nsched_ctxs2;
		sc_hypervisor_get_ctxs_on_level(&sched_ctxs2, &nsched_ctxs2, 0, STARPU_NMAX_SCHED_CTXS);

		if(nsched_ctxs2  > 0)
		{
			_update_max_hierarchically(sched_ctxs2, nsched_ctxs2);
			int s;
			int current_total_max_nworkers = 0;
			double max_nflops = 0.0;
			unsigned max_nflops_sched_ctx = sched_ctxs2[0];
			for(s = 0; s < nsched_ctxs2; s++)
			{
				struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctxs2[s]);
				current_total_max_nworkers += config->max_nworkers;
				if(max_nflops < hypervisor.sched_ctx_w[sched_ctxs2[s]].remaining_flops)
				{
					max_nflops = hypervisor.sched_ctx_w[sched_ctxs2[s]].remaining_flops;
					max_nflops_sched_ctx = sched_ctxs2[s];
				}
			}

			int max_possible_workers = starpu_worker_get_count();
			/*if the sum of the max cpus is smaller than the total cpus available
			  increase the max for the ones having more ready tasks to exec */
			if(current_total_max_nworkers < max_possible_workers)
			{
				int diff = max_possible_workers - current_total_max_nworkers;
				struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(max_nflops_sched_ctx);
				config->max_nworkers += diff;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
				printf("%u: redib max_nworkers incr %d \n",  max_nflops_sched_ctx, config->max_nworkers);
#endif
				_update_max_diff_hierarchically(max_nflops_sched_ctx, diff);
			}
			free(sched_ctxs2);
		}
	}



	/*if the sum of the max cpus is smaller than the total cpus available
	  increase the max for the ones having more ready tasks to exec */
	/* if(configured && total_max_nworkers < max_workers) */
/* 	{ */
/* 		int diff = max_workers - total_max_nworkers; */
/* 		int max_nready = -1; */
/* 		unsigned max_nready_sched_ctx = sched_ctxs[0]; */
/* 		for(i = 0; i < nsched_ctxs; i++) */
/* 		{ */
/* 			int nready_tasks = starpu_sched_ctx_get_nready_tasks(sched_ctxs[i]); */
/* 			if(max_nready < nready_tasks) */
/* 			{ */
/* 				max_nready = nready_tasks; */
/* 				max_nready_sched_ctx = sched_ctxs[i]; */
/* 			} */
/* 		} */
/* 		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(max_nready_sched_ctx); */
/* 		config->max_nworkers += diff; */
/* 		printf("%d: redib max_nworkers incr %d \n",  max_nready_sched_ctx, config->max_nworkers); */
/* 	} */

}

/* notifies the hypervisor that a new task was pushed on the queue of the worker */
static void notify_pushed_task(unsigned sched_ctx, int worker)
{
	hypervisor.sched_ctx_w[sched_ctx].pushed_tasks[worker]++;
	if(hypervisor.sched_ctx_w[sched_ctx].total_flops != 0.0 && hypervisor.sched_ctx_w[sched_ctx].start_time == 0.0)
		hypervisor.sched_ctx_w[sched_ctx].start_time = starpu_timing_now();

	if(hypervisor.sched_ctx_w[sched_ctx].total_flops != 0.0 && hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker] == 0.0)
	{
		hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker] = starpu_timing_now();
	}

	int ntasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].pushed_tasks);

	if((hypervisor.min_tasks == 0 || (!(hypervisor.resize[sched_ctx] == 0 && imposed_resize) && ntasks == hypervisor.min_tasks)) && hypervisor.check_min_tasks[sched_ctx])
	{
		hypervisor.resize[sched_ctx] = 1;
		if(imposed_resize) imposed_resize = 0;
		hypervisor.check_min_tasks[sched_ctx] = 0;
	}

	if(hypervisor.policy.handle_pushed_task)
		hypervisor.policy.handle_pushed_task(sched_ctx, worker);
}

unsigned choose_ctx_to_steal(int worker)
{
	int j;
	int ns = hypervisor.nsched_ctxs;
	int max_ready_tasks = 0;
	unsigned chosen_ctx = STARPU_NMAX_SCHED_CTXS;
	for(j = 0; j < ns; j++)
	{
		unsigned other_ctx = hypervisor.sched_ctxs[j];
		int nready = starpu_sched_ctx_get_nready_tasks(other_ctx);
		if(!starpu_sched_ctx_contains_worker(worker, other_ctx) && max_ready_tasks < nready)
		{
			max_ready_tasks = nready;
			chosen_ctx = other_ctx;
		}
	}
	return chosen_ctx;
}

/* notifies the hypervisor that the worker spent another cycle in idle time */
static void notify_idle_cycle(unsigned sched_ctx, int worker, double idle_time)
{
	if(hypervisor.start_executing_time == 0.0) return;
	struct sc_hypervisor_wrapper *sc_w = &hypervisor.sched_ctx_w[sched_ctx];
	sc_w->current_idle_time[worker] += idle_time;

	if(sc_w->idle_start_time[worker] == 0.0 && sc_w->hyp_react_start_time != 0.0)
		sc_w->idle_start_time[worker] = starpu_timing_now();


	if(sc_w->idle_start_time[worker] > 0.0)
	{
		double end_time  = starpu_timing_now();
		sc_w->idle_time[worker] += (end_time - sc_w->idle_start_time[worker]) / 1000000.0; /* in seconds */
	}

	hypervisor.sched_ctx_w[sched_ctx].idle_start_time[worker] = starpu_timing_now();

	if(hypervisor.resize[sched_ctx] && hypervisor.policy.handle_idle_cycle)
	{
		if(sc_w->hyp_react_start_time == 0.0)
			sc_hypervisor_reset_react_start_time(sched_ctx, 1);

		double curr_time = starpu_timing_now();
		double elapsed_time = (curr_time - sc_w->hyp_react_start_time) / 1000000.0; /* in seconds */
		if(sc_w->sched_ctx != STARPU_NMAX_SCHED_CTXS && elapsed_time > sc_w->config->time_sample)
		{
			unsigned idle_everywhere = 0;
			unsigned *sched_ctxs = NULL;
			unsigned nsched_ctxs = 0;
			int ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
			if(ret != EBUSY)
			{
				if(sc_hypervisor_check_idle(sched_ctx, worker))
				{
					idle_everywhere = 1;

					nsched_ctxs = starpu_worker_get_sched_ctx_list(worker, &sched_ctxs);
					int s;
					for(s = 0; s < nsched_ctxs; s++)
					{
						if(hypervisor.sched_ctx_w[sched_ctxs[s]].sched_ctx != STARPU_NMAX_SCHED_CTXS)
						{
							if(!sc_hypervisor_check_idle(sched_ctxs[s], worker))
								idle_everywhere = 0;
						}
					}
					free(sched_ctxs);
				}
				STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
			}

			if(idle_everywhere)
			{
				double hyp_overhead_start = starpu_timing_now();
				if(elapsed_time > (sc_w->config->time_sample*2))
					hypervisor.policy.handle_idle_cycle(sched_ctx, worker);
				double hyp_overhead_end = starpu_timing_now();
				hyp_overhead += (hyp_overhead_end - hyp_overhead_start);
				if(elapsed_time > (sc_w->config->time_sample*2))
					sc_hypervisor_reset_react_start_time(sched_ctx, 1);
				else
					sc_hypervisor_reset_react_start_time(sched_ctx, 0);
			}
		}
	}
	return;
}

void _update_real_start_time_hierarchically(unsigned sched_ctx)
{
	hypervisor.sched_ctx_w[sched_ctx].real_start_time = starpu_timing_now();
	if(starpu_sched_ctx_get_hierarchy_level(sched_ctx) > 0)
	{
		_update_real_start_time_hierarchically(starpu_sched_ctx_get_inheritor(sched_ctx));
	}
	return;
}

/* notifies the hypervisor that the worker is no longer idle and a new task was pushed on its queue */
static void notify_poped_task(unsigned sched_ctx, int worker)
{
	if(hypervisor.start_executing_time == 0.0)
		hypervisor.start_executing_time = starpu_timing_now();
	if(!hypervisor.resize[sched_ctx])
		hypervisor.resize[sched_ctx] = 1;

	if(hypervisor.sched_ctx_w[sched_ctx].total_flops != 0.0 && hypervisor.sched_ctx_w[sched_ctx].real_start_time == 0.0)
		_update_real_start_time_hierarchically(sched_ctx);

	if(hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker] == 0.0)
	{
		hypervisor.sched_ctx_w[sched_ctx].start_time_w[worker] = starpu_timing_now();
	}

	hypervisor.sched_ctx_w[sched_ctx].exec_start_time[worker] = starpu_timing_now();

	if(hypervisor.sched_ctx_w[sched_ctx].idle_start_time[worker] > 0.0)
	{
		int ns = hypervisor.nsched_ctxs;
		int j;
		for(j = 0; j < ns; j++)
		{
			if(hypervisor.sched_ctxs[j] != sched_ctx)
			{
				if(hypervisor.sched_ctx_w[hypervisor.sched_ctxs[j]].idle_start_time[worker] > 0.0)
					hypervisor.sched_ctx_w[hypervisor.sched_ctxs[j]].compute_partial_idle[worker] = 1;
			}
		}
		double end_time  = starpu_timing_now();
		double idle = (end_time - hypervisor.sched_ctx_w[sched_ctx].idle_start_time[worker]) / 1000000.0; /* in seconds */

		if(hypervisor.sched_ctx_w[sched_ctx].compute_partial_idle[worker])
			hypervisor.sched_ctx_w[sched_ctx].idle_time[worker] += idle / 2.0;
		else
			hypervisor.sched_ctx_w[sched_ctx].idle_time[worker] += idle;

		hypervisor.sched_ctx_w[sched_ctx].compute_partial_idle[worker] = 0;
		hypervisor.sched_ctx_w[sched_ctx].idle_start_time[worker] = 0.0;
	}

	if(hypervisor.resize[sched_ctx])
		hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker] = 0.0;

	if(hypervisor.policy.handle_idle_end)
		hypervisor.policy.handle_idle_end(sched_ctx, worker);
}


static void _update_counters_hierarchically(int worker, unsigned sched_ctx, double flops, size_t data_size)
{
	hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker]++;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_flops[worker] += flops;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_data[worker] += data_size ;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_tasks[worker]++ ;
	hypervisor.sched_ctx_w[sched_ctx].total_elapsed_flops[worker] += flops;

	STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
	hypervisor.sched_ctx_w[sched_ctx].remaining_flops -= flops;
	STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);

	if(starpu_sched_ctx_get_hierarchy_level(sched_ctx) > 0)
		_update_counters_hierarchically(worker, starpu_sched_ctx_get_inheritor(sched_ctx), flops, data_size);

	return;
}

/* notifies the hypervisor that a tagged task has just been executed */
static void notify_post_exec_task(struct starpu_task *task, size_t data_size, uint32_t footprint, int task_tag, double flops)
{
	unsigned sched_ctx = task->sched_ctx;
	int worker = starpu_worker_get_id_check();

	if(hypervisor.sched_ctx_w[sched_ctx].exec_start_time[worker] != 0.0)
	{
		double current_time = starpu_timing_now();
		double exec_time = (current_time -
				    hypervisor.sched_ctx_w[sched_ctx].exec_start_time[worker]) / 1000000.0; /* in seconds */
		hypervisor.sched_ctx_w[sched_ctx].exec_time[worker] += exec_time;
		hypervisor.sched_ctx_w[sched_ctx].exec_start_time[worker] = 0.0;
	}

	hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker]++;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_flops[worker] += flops;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_data[worker] += data_size ;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_tasks[worker]++ ;
	hypervisor.sched_ctx_w[sched_ctx].total_elapsed_flops[worker] += flops;

	STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
	hypervisor.sched_ctx_w[sched_ctx].remaining_flops -= flops;
	STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);

	if(_sc_hypervisor_use_lazy_resize())
		_ack_resize_completed(sched_ctx, worker);

	if(starpu_sched_ctx_get_hierarchy_level(sched_ctx) > 0)
	{
		_update_counters_hierarchically(worker, starpu_sched_ctx_get_inheritor(sched_ctx), flops, data_size);
	}

	if(hypervisor.resize[sched_ctx])
	{
		if(hypervisor.policy.handle_poped_task)
		{
			if(hypervisor.sched_ctx_w[sched_ctx].hyp_react_start_time == 0.0)
				sc_hypervisor_reset_react_start_time(sched_ctx, 1);

			double curr_time = starpu_timing_now();
			double elapsed_time = (curr_time - hypervisor.sched_ctx_w[sched_ctx].hyp_react_start_time) / 1000000.0; /* in seconds */
			if(hypervisor.sched_ctx_w[sched_ctx].sched_ctx != STARPU_NMAX_SCHED_CTXS && elapsed_time > hypervisor.sched_ctx_w[sched_ctx].config->time_sample)
			{
				double hyp_overhead_start = starpu_timing_now();
				if(elapsed_time > (hypervisor.sched_ctx_w[sched_ctx].config->time_sample*2))
					hypervisor.policy.handle_poped_task(sched_ctx, worker, task, footprint);
				double hyp_overhead_end = starpu_timing_now();
				hyp_overhead += (hyp_overhead_end - hyp_overhead_start);
				if(elapsed_time > (hypervisor.sched_ctx_w[sched_ctx].config->time_sample*2))
					sc_hypervisor_reset_react_start_time(sched_ctx, 1);
				else
					sc_hypervisor_reset_react_start_time(sched_ctx, 0);
			}
			else
                                /* no need to consider resizing, just remove the task from the pool if the strategy requires it*/
				hypervisor.policy.handle_poped_task(sched_ctx, -2, task, footprint);
		}
	}
/* 	STARPU_PTHREAD_MUTEX_LOCK(&act_hypervisor_mutex); */
/* 	_ack_resize_completed(sched_ctx, worker); */
/* 	STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex); */
	if(hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker] % 200 == 0)
		_print_current_time();

	if(task_tag <= 0)
		return;

	unsigned conf_sched_ctx;
	unsigned i;
	unsigned ns = hypervisor.nsched_ctxs;

	for(i = 0; i < ns; i++)
	{
		struct configuration_entry *entry;

		conf_sched_ctx = hypervisor.sched_ctxs[i];
		STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.conf_mut[conf_sched_ctx]);

		HASH_FIND_INT(hypervisor.configurations[conf_sched_ctx], &task_tag, entry);

		if (entry != NULL)
		{
			struct sc_hypervisor_policy_config *config = entry->configuration;

			sc_hypervisor_set_config(conf_sched_ctx, config);
			HASH_DEL(hypervisor.configurations[conf_sched_ctx], entry);
			free(config);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.conf_mut[conf_sched_ctx]);
	}

	if(hypervisor.resize[sched_ctx])
	{
		STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.resize_mut[sched_ctx]);

		if(hypervisor.policy.handle_post_exec_hook)
		{
			/* Check whether 'task_tag' is in the 'resize_requests' set.  */
			struct resize_request_entry *entry;
			HASH_FIND_INT(hypervisor.resize_requests[sched_ctx], &task_tag, entry);
			if (entry != NULL)
			{
				hypervisor.policy.handle_post_exec_hook(sched_ctx, task_tag);
				HASH_DEL(hypervisor.resize_requests[sched_ctx], entry);
				free(entry);
			}

		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.resize_mut[sched_ctx]);
	}
	return;
}

static void notify_submitted_job(struct starpu_task *task, uint32_t footprint, size_t data_size)
{
	unsigned sched_ctx = task->sched_ctx;
	STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
	hypervisor.sched_ctx_w[sched_ctx].submitted_flops += task->flops;
	STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);

	/* signaled by the user - no need to wait for them */
	/* if(hypervisor.policy.handle_submitted_job && !type_of_tasks_known) */
	/* 	hypervisor.policy.handle_submitted_job(task->cl, task->sched_ctx, footprint, data_size); */
}

static void notify_empty_ctx(unsigned sched_ctx_id, struct starpu_task *task)
{
	sc_hypervisor_resize_ctxs(NULL, -1 , NULL, -1);
}

void sc_hypervisor_set_type_of_task(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint, size_t data_size)
{
	type_of_tasks_known = 1;
	if(hypervisor.policy.handle_submitted_job)
		hypervisor.policy.handle_submitted_job(cl, sched_ctx, footprint, data_size);
}

static void notify_delete_context(unsigned sched_ctx)
{
	_print_current_time();
	sc_hypervisor_unregister_ctx(sched_ctx);
}

void sc_hypervisor_size_ctxs(unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers)
{
//	STARPU_PTHREAD_MUTEX_LOCK(&act_hypervisor_mutex);
	unsigned curr_nsched_ctxs = sched_ctxs == NULL ? hypervisor.nsched_ctxs : (unsigned)nsched_ctxs;
	unsigned *curr_sched_ctxs = sched_ctxs == NULL ? hypervisor.sched_ctxs : sched_ctxs;
//	STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
	unsigned s;
	for(s = 0; s < curr_nsched_ctxs; s++)
		hypervisor.resize[curr_sched_ctxs[s]] = 1;

	if(hypervisor.policy.size_ctxs)
		hypervisor.policy.size_ctxs(curr_sched_ctxs, curr_nsched_ctxs, workers, nworkers);
}

struct sc_hypervisor_wrapper* sc_hypervisor_get_wrapper(unsigned sched_ctx)
{
	return &hypervisor.sched_ctx_w[sched_ctx];
}

unsigned* sc_hypervisor_get_sched_ctxs()
{
	return hypervisor.sched_ctxs;
}

int sc_hypervisor_get_nsched_ctxs()
{
	int ns;
	ns = hypervisor.nsched_ctxs;
	return ns;
}

int _sc_hypervisor_use_lazy_resize(void)
{
	char* lazy = getenv("SC_HYPERVISOR_LAZY_RESIZE");
	return lazy ? atoi(lazy)  : 1;
}

void sc_hypervisor_save_size_req(unsigned *sched_ctxs, int nsched_ctxs, int *workers, int nworkers)
{
	hypervisor.sr = (struct size_request*)malloc(sizeof(struct size_request));
	hypervisor.sr->sched_ctxs = sched_ctxs;
	hypervisor.sr->nsched_ctxs = nsched_ctxs;
	hypervisor.sr->workers = workers;
	hypervisor.sr->nworkers = nworkers;
}

unsigned sc_hypervisor_get_size_req(unsigned **sched_ctxs, int* nsched_ctxs, int **workers, int *nworkers)
{
	if(hypervisor.sr != NULL)
	{
		*sched_ctxs = hypervisor.sr->sched_ctxs;
		*nsched_ctxs = hypervisor.sr->nsched_ctxs;
		*workers = hypervisor.sr->workers;
		*nworkers = hypervisor.sr->nworkers;
		return 1;
	}
	return 0;
}

void sc_hypervisor_free_size_req(void)
{
	if(hypervisor.sr != NULL)
	{
		free(hypervisor.sr);
		hypervisor.sr = NULL;
	}
}

double _get_optimal_v(unsigned sched_ctx)
{
	return hypervisor.optimal_v[sched_ctx];
}

void _set_optimal_v(unsigned sched_ctx, double optimal_v)
{
	hypervisor.optimal_v[sched_ctx] = optimal_v;
}

static struct types_of_workers* _init_structure_types_of_workers(void)
{
	struct types_of_workers *tw = (struct types_of_workers*)malloc(sizeof(struct types_of_workers));
        tw->ncpus = 0;
	tw->ncuda = 0;
        tw->nw = 0;
        return tw;
}

struct types_of_workers* sc_hypervisor_get_types_of_workers(int *workers, unsigned nworkers)
{
	struct types_of_workers *tw = _init_structure_types_of_workers();

        unsigned w;
	for(w = 0; w < nworkers; w++)
        {
                enum starpu_worker_archtype arch = workers == NULL ? starpu_worker_get_type((int)w) : starpu_worker_get_type(workers[w]);
                if(arch == STARPU_CPU_WORKER)
			tw->ncpus++;
                if(arch == STARPU_CUDA_WORKER)
			tw->ncuda++;
        }
        if(tw->ncpus > 0) tw->nw++;
        if(tw->ncuda > 0) tw->nw++;
	return tw;
}

void sc_hypervisor_update_diff_total_flops(unsigned sched_ctx, double diff_total_flops)
{
//	double hyp_overhead_start = starpu_timing_now();
	STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
	hypervisor.sched_ctx_w[sched_ctx].total_flops += diff_total_flops;
	hypervisor.sched_ctx_w[sched_ctx].remaining_flops += diff_total_flops;
	STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
/* 	double hyp_overhead_end = starpu_timing_now(); */
/* 	hyp_overhead += (hyp_overhead_end - hyp_overhead_start); */
	if(starpu_sched_ctx_get_hierarchy_level(sched_ctx) > 0)
		sc_hypervisor_update_diff_total_flops(starpu_sched_ctx_get_inheritor(sched_ctx), diff_total_flops);
	return;

}

void sc_hypervisor_update_diff_elapsed_flops(unsigned sched_ctx, double diff_elapsed_flops)
{
//	double hyp_overhead_start = starpu_timing_now();
	int workerid = starpu_worker_get_id();
	if(workerid != -1)
	{
//		STARPU_PTHREAD_MUTEX_LOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
		hypervisor.sched_ctx_w[sched_ctx].elapsed_flops[workerid] += diff_elapsed_flops;
		hypervisor.sched_ctx_w[sched_ctx].total_elapsed_flops[workerid] += diff_elapsed_flops;
//		STARPU_PTHREAD_MUTEX_UNLOCK(&hypervisor.sched_ctx_w[sched_ctx].mutex);
	}
/* 	double hyp_overhead_end = starpu_timing_now(); */
/* 	hyp_overhead += (hyp_overhead_end - hyp_overhead_start); */
	if(starpu_sched_ctx_get_hierarchy_level(sched_ctx) > 0)
		sc_hypervisor_update_diff_elapsed_flops(starpu_sched_ctx_get_inheritor(sched_ctx), diff_elapsed_flops);
	return;
}

void sc_hypervisor_get_ctxs_on_level(unsigned **sched_ctxs, int *nsched_ctxs, unsigned hierarchy_level, unsigned father_sched_ctx_id)
{
	unsigned s;
	*nsched_ctxs = 0;
	*sched_ctxs = (unsigned*)malloc(hypervisor.nsched_ctxs * sizeof(unsigned));
	for(s = 0; s < hypervisor.nsched_ctxs; s++)
	{
		/* if father == STARPU_NMAX_SCHED_CTXS we take all the ctxs in this level */
		if(starpu_sched_ctx_get_hierarchy_level(hypervisor.sched_ctxs[s]) == hierarchy_level &&
		   (starpu_sched_ctx_get_inheritor(hypervisor.sched_ctxs[s]) == father_sched_ctx_id || father_sched_ctx_id == STARPU_NMAX_SCHED_CTXS))
		        (*sched_ctxs)[(*nsched_ctxs)++] = hypervisor.sched_ctxs[s];
	}
	if(*nsched_ctxs == 0)
	{
		free(*sched_ctxs);
		*sched_ctxs = NULL;
	}
	return;
}

unsigned sc_hypervisor_get_nhierarchy_levels(void)
{
	unsigned nlevels = 0;
	unsigned level = 0;
	unsigned levels[STARPU_NMAX_SCHED_CTXS];
	unsigned s, l;
	for(s = 0; s < hypervisor.nsched_ctxs; s++)
	{
		level = starpu_sched_ctx_get_hierarchy_level(hypervisor.sched_ctxs[s]);
		unsigned found = 0;
		for(l = 0; l < nlevels; l++)
			if(levels[l] == level)
				found = 1;
		if(!found)
			levels[nlevels++] = level;
	}
	return nlevels;
}

void sc_hypervisor_get_leaves(unsigned *sched_ctxs, int nsched_ctxs, unsigned *leaves, int *nleaves)
{
	int s, s2;
	for(s = 0; s < nsched_ctxs; s++)
	{
		unsigned is_someones_father = 0;
		for(s2 = 0; s2 < nsched_ctxs; s2++)
		{
			unsigned father = starpu_sched_ctx_get_inheritor(sched_ctxs[s2]);
			if(sched_ctxs[s] == father)
			{
				is_someones_father = 1;
				break;
			}
		}
		if(!is_someones_father)
			leaves[(*nleaves)++] = sched_ctxs[s];
	}
	return;
}


void sc_hypervisor_init_worker(int workerid, unsigned sched_ctx)
{
	if(hypervisor.policy.init_worker)
                hypervisor.policy.init_worker(workerid, sched_ctx);
}
