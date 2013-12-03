/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012  INRIA
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
#include <common/uthash.h>
#include <starpu_config.h>

unsigned imposed_resize = 0;
unsigned type_of_tasks_known = 0;
struct starpu_sched_ctx_performance_counters* perf_counters = NULL;

static void notify_idle_cycle(unsigned sched_ctx, int worker, double idle_time);
static void notify_pushed_task(unsigned sched_ctx, int worker);
static void notify_poped_task(unsigned sched_ctx, int worker, struct starpu_task *task, size_t data_size, uint32_t footprint);
static void notify_post_exec_hook(unsigned sched_ctx, int taskid);
static void notify_idle_end(unsigned sched_ctx, int  worker);
static void notify_submitted_job(struct starpu_task *task, unsigned footprint);
static void notify_delete_context(unsigned sched_ctx);

extern struct sc_hypervisor_policy idle_policy;
extern struct sc_hypervisor_policy app_driven_policy;
extern struct sc_hypervisor_policy gflops_rate_policy;
#ifdef STARPU_HAVE_GLPK_H
extern struct sc_hypervisor_policy feft_lp_policy;
extern struct sc_hypervisor_policy teft_lp_policy;
extern struct sc_hypervisor_policy ispeed_lp_policy;
extern struct sc_hypervisor_policy debit_lp_policy;
#endif // STARPU_HAVE_GLPK_
extern struct sc_hypervisor_policy ispeed_policy;


static struct sc_hypervisor_policy *predefined_policies[] =
{
        &idle_policy,
	&app_driven_policy,
#ifdef STARPU_HAVE_GLPK_H
	&feft_lp_policy,
	&teft_lp_policy,
	&ispeed_lp_policy,
	&debit_lp_policy,
#endif // STARPU_HAVE_GLPK_H
	&gflops_rate_policy,
	&ispeed_policy
};

static void _load_hypervisor_policy(struct sc_hypervisor_policy *policy)
{
	STARPU_ASSERT(policy);

	hypervisor.policy.name = policy->name;
	hypervisor.policy.size_ctxs = policy->size_ctxs;
	hypervisor.policy.handle_poped_task = policy->handle_poped_task;
	hypervisor.policy.handle_pushed_task = policy->handle_pushed_task;
	hypervisor.policy.handle_idle_cycle = policy->handle_idle_cycle;
	hypervisor.policy.handle_idle_end = policy->handle_idle_end;
	hypervisor.policy.handle_post_exec_hook = policy->handle_post_exec_hook;
	hypervisor.policy.handle_submitted_job = policy->handle_submitted_job;
	hypervisor.policy.end_ctx = policy->end_ctx;
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
		policy_name = getenv("HYPERVISOR_POLICY");
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
struct starpu_sched_ctx_performance_counters* sc_hypervisor_init(struct sc_hypervisor_policy *hypervisor_policy)
{
	hypervisor.min_tasks = 0;
	hypervisor.nsched_ctxs = 0;
	char* vel_gap = getenv("MAX_VELOCITY_GAP");
	hypervisor.max_velocity_gap = vel_gap ? atof(vel_gap) : SC_VELOCITY_MAX_GAP_DEFAULT;
	char* crit =  getenv("HYPERVISOR_TRIGGER_RESIZE");
	hypervisor.resize_criteria = strcmp(crit,"idle") == 0 ? SC_IDLE : (strcmp(crit,"speed") == 0 ? SC_VELOCITY : SC_NOTHING);

	starpu_pthread_mutex_init(&act_hypervisor_mutex, NULL);
	hypervisor.start_executing_time = starpu_timing_now();
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
		hypervisor.sched_ctx_w[i].resize_ack.receiver_sched_ctx = -1;
		hypervisor.sched_ctx_w[i].resize_ack.moved_workers = NULL;
		hypervisor.sched_ctx_w[i].resize_ack.nmoved_workers = 0;
		hypervisor.sched_ctx_w[i].resize_ack.acked_workers = NULL;
		starpu_pthread_mutex_init(&hypervisor.sched_ctx_w[i].mutex, NULL);

		int j;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
		{
			hypervisor.sched_ctx_w[i].current_idle_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].pushed_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].poped_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].elapsed_flops[j] = 0.0;
			hypervisor.sched_ctx_w[i].elapsed_data[j] = 0;
			hypervisor.sched_ctx_w[i].elapsed_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].total_elapsed_flops[j] = 0.0;
			hypervisor.sched_ctx_w[i].worker_to_be_removed[j] = 0;
			hypervisor.sched_ctx_w[i].ref_velocity[j] = -1.0;
		}
	}

	struct sc_hypervisor_policy *selected_hypervisor_policy = _select_hypervisor_policy(hypervisor_policy);
	_load_hypervisor_policy(selected_hypervisor_policy);

	perf_counters = (struct starpu_sched_ctx_performance_counters*)malloc(sizeof(struct starpu_sched_ctx_performance_counters));
	perf_counters->notify_idle_cycle = notify_idle_cycle;
	perf_counters->notify_pushed_task = notify_pushed_task;
	perf_counters->notify_poped_task = notify_poped_task;
	perf_counters->notify_post_exec_hook = notify_post_exec_hook;
	perf_counters->notify_idle_end = notify_idle_end;
	perf_counters->notify_submitted_job = notify_submitted_job;
	perf_counters->notify_delete_context = notify_delete_context;

	starpu_sched_ctx_notify_hypervisor_exists();

	return perf_counters;
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
	if(!getenv("HYPERVISOR_STOP_PRINT"))
	{
		double curr_time = starpu_timing_now();
		double elapsed_time = (curr_time - hypervisor.start_executing_time) / 1000000.0; /* in seconds */
		fprintf(stdout, "Time: %lf\n", elapsed_time);
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS)
			{
				struct sc_hypervisor_wrapper *sc_w = &hypervisor.sched_ctx_w[hypervisor.sched_ctxs[i]];
				
				double cpu_speed = sc_hypervisor_get_velocity(sc_w, STARPU_CPU_WORKER);
				double cuda_speed = sc_hypervisor_get_velocity(sc_w, STARPU_CUDA_WORKER);
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
//	printf("shutdown\n");
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
                if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && hypervisor.nsched_ctxs > 0)
		{
			sc_hypervisor_stop_resize(hypervisor.sched_ctxs[i]);
			sc_hypervisor_unregister_ctx(hypervisor.sched_ctxs[i]);
			starpu_pthread_mutex_destroy(&hypervisor.sched_ctx_w[i].mutex);
		}
	}
	perf_counters->notify_idle_cycle = NULL;
	perf_counters->notify_pushed_task = NULL;
	perf_counters->notify_poped_task = NULL;
	perf_counters->notify_post_exec_hook = NULL;
	perf_counters->notify_idle_end = NULL;
	perf_counters->notify_delete_context = NULL;

	free(perf_counters);
	perf_counters = NULL;

	starpu_pthread_mutex_destroy(&act_hypervisor_mutex);
}

/* the hypervisor is in charge only of the contexts registered to it*/
void sc_hypervisor_register_ctx(unsigned sched_ctx, double total_flops)
{
	starpu_pthread_mutex_lock(&act_hypervisor_mutex);
	hypervisor.configurations[sched_ctx] = NULL;
	hypervisor.resize_requests[sched_ctx] = NULL;
	starpu_pthread_mutex_init(&hypervisor.conf_mut[sched_ctx], NULL);
	starpu_pthread_mutex_init(&hypervisor.resize_mut[sched_ctx], NULL);

	_add_config(sched_ctx);
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = sched_ctx;
	hypervisor.sched_ctxs[hypervisor.nsched_ctxs++] = sched_ctx;

	hypervisor.sched_ctx_w[sched_ctx].total_flops = total_flops;
	hypervisor.sched_ctx_w[sched_ctx].remaining_flops = total_flops;
	if(strcmp(hypervisor.policy.name, "app_driven") == 0)
		hypervisor.resize[sched_ctx] = 1;
	starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
}

static int _get_first_free_sched_ctx(int *sched_ctxs, int nsched_ctxs)
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

/* unregistered contexts will no longer be resized */
void sc_hypervisor_unregister_ctx(unsigned sched_ctx)
{
	if(hypervisor.policy.end_ctx)
		hypervisor.policy.end_ctx(sched_ctx);
	starpu_pthread_mutex_lock(&act_hypervisor_mutex);
	unsigned i;
	for(i = 0; i < hypervisor.nsched_ctxs; i++)
	{
		if(hypervisor.sched_ctxs[i] == (int)sched_ctx)
		{
			hypervisor.sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
			break;
		}
	}

	_rearange_sched_ctxs(hypervisor.sched_ctxs, hypervisor.nsched_ctxs);
	hypervisor.nsched_ctxs--;
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = STARPU_NMAX_SCHED_CTXS;
	_remove_config(sched_ctx);

/* 	free(hypervisor.configurations[sched_ctx]); */
/* 	free(hypervisor.resize_requests[sched_ctx]); */
	starpu_pthread_mutex_destroy(&hypervisor.conf_mut[sched_ctx]);
	starpu_pthread_mutex_destroy(&hypervisor.resize_mut[sched_ctx]);
	if(hypervisor.nsched_ctxs == 1)
		sc_hypervisor_stop_resize(hypervisor.sched_ctxs[0]);

	starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
}

static double _get_best_total_elapsed_flops(struct sc_hypervisor_wrapper* sc_w, int *npus, enum starpu_worker_archtype req_arch)
{
	double ret_val = 0.0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
        int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
                workers->init_iterator(workers, &it);

        while(workers->has_next(workers, &it))
	{
                worker = workers->get_next(workers, &it);
                enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
                if(arch == req_arch)
                {
			if(sc_w->total_elapsed_flops[worker] > ret_val)
				ret_val = sc_w->total_elapsed_flops[worker];
			(*npus)++;
                }
        }

	return ret_val;
}

double _get_max_velocity_gap()
{
	return hypervisor.max_velocity_gap;
}

unsigned sc_hypervisor_get_resize_criteria()
{
	return hypervisor.resize_criteria;
}

/* compute an average value of the cpu/cuda velocity */
double sc_hypervisorsc_hypervisor_get_velocity_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch)
{
        int npus = 0;
        double elapsed_flops = _get_best_total_elapsed_flops(sc_w, &npus, arch) / 1000000000.0 ; /* in gflops */
	if(npus == 0)
		return -1.0; 

        if( elapsed_flops != 0.0)
        {
                double curr_time = starpu_timing_now();
                double elapsed_time = (curr_time - sc_w->real_start_time) / 1000000.0; /* in seconds */
		double velocity = (elapsed_flops/elapsed_time); /* in Gflops/s */
                return velocity;
        }

        return -1.0;
}

/* compute an average value of the cpu/cuda old velocity */
double sc_hypervisor_get_ref_velocity_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch)
{
	double ref_velocity = 0.0;
	unsigned nw = 0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		if(sc_w->ref_velocity[worker] > 1.0)
		{
			ref_velocity += sc_w->ref_velocity[worker];
			nw++;
		}
	}
	
	if(nw > 0)
		return ref_velocity / nw;
	return -1.0;
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


static void _get_cpus(int *workers, int nworkers, int *cpus, int *ncpus)
{
	int i, worker;
	*ncpus = 0;

	for(i = 0; i < nworkers; i++)
	{
		worker = workers[i];
		enum starpu_worker_archtype arch = starpu_worker_get_type(worker);
		if(arch == STARPU_CPU_WORKER)
			cpus[(*ncpus)++] = worker;
	}
}

int sc_hypervisor_get_nworkers_ctx(unsigned sched_ctx, enum starpu_worker_archtype arch)
{
	int nworkers_ctx = 0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
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
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		ret_val += sc_w->elapsed_flops[i];
	return ret_val;
}

double sc_hypervisor_get_total_elapsed_flops_per_sched_ctx(struct sc_hypervisor_wrapper* sc_w)
{
	double ret_val = 0.0;
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		ret_val += sc_w->total_elapsed_flops[i];
	return ret_val;
}


void _reset_resize_sample_info(unsigned sender_sched_ctx, unsigned receiver_sched_ctx)
{
	/* info concerning only the gflops_rate strateg */
	struct sc_hypervisor_wrapper *sender_sc_w = &hypervisor.sched_ctx_w[sender_sched_ctx];
	struct sc_hypervisor_wrapper *receiver_sc_w = &hypervisor.sched_ctx_w[receiver_sched_ctx];
	
	double start_time =  starpu_timing_now();
	sender_sc_w->start_time = start_time;
	_set_elapsed_flops_per_sched_ctx(sender_sched_ctx, 0.0);

	receiver_sc_w->start_time = start_time;
	_set_elapsed_flops_per_sched_ctx(receiver_sched_ctx, 0.0);
}

/* actually move the workers: the cpus are moved, gpus are only shared  */
/* forbids another resize request before this one is take into account */
void sc_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move, unsigned now)
{
	if(nworkers_to_move > 0 && hypervisor.resize[sender_sched_ctx])
	{
		_print_current_time();
		unsigned j;
		printf("resize ctx %d with %d workers", sender_sched_ctx, nworkers_to_move);
		for(j = 0; j < nworkers_to_move; j++)
			printf(" %d", workers_to_move[j]);
		printf("\n");
		starpu_trace_user_event(1);
		hypervisor.allow_remove[receiver_sched_ctx] = 0;
		starpu_sched_ctx_add_workers(workers_to_move, nworkers_to_move, receiver_sched_ctx);

		if(now)
		{
			unsigned j;
			printf("remove now from ctx %d:", sender_sched_ctx);
			for(j = 0; j < nworkers_to_move; j++)
				printf(" %d", workers_to_move[j]);
			printf("\n");

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

				starpu_pthread_mutex_unlock(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
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
		printf("add to ctx %d:", sched_ctx);
		for(j = 0; j < nworkers_to_add; j++)
			printf(" %d", workers_to_add[j]);
		printf("\n");
		starpu_sched_ctx_add_workers(workers_to_add, nworkers_to_add, sched_ctx);
		struct sc_hypervisor_policy_config *new_config = sc_hypervisor_get_config(sched_ctx);
		unsigned i;
		for(i = 0; i < nworkers_to_add; i++)
			new_config->max_idle[workers_to_add[i]] = new_config->max_idle[workers_to_add[i]] != MAX_IDLE_TIME ? new_config->max_idle[workers_to_add[i]] :  new_config->new_workers_max_idle;

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
			printf("remove explicitley now from ctx %d:", sched_ctx);
			for(j = 0; j < nworkers_to_remove; j++)
				printf(" %d", workers_to_remove[j]);
			printf("\n");
			
			starpu_sched_ctx_remove_workers(workers_to_remove, nworkers_to_remove, sched_ctx);
		}
		else
		{
			printf("try to remove from ctx %d: ", sched_ctx);
			unsigned j;
			for(j = 0; j < nworkers_to_remove; j++)
				printf(" %d", workers_to_remove[j]);
			printf("\n");

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
				starpu_pthread_mutex_unlock(&hypervisor.sched_ctx_w[sched_ctx].mutex);
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
			starpu_pthread_mutex_lock(&sc_w->mutex);
			unsigned only_remove = 0;
			if(sc_w->resize_ack.receiver_sched_ctx == -1 && hypervisor.sched_ctxs[i] != (int)sched_ctx &&
			   sc_w->resize_ack.nmoved_workers > 0 && starpu_sched_ctx_contains_worker(worker, hypervisor.sched_ctxs[i]))
			{
				int j;
				for(j = 0; j < sc_w->resize_ack.nmoved_workers; j++)
					if(sc_w->resize_ack.moved_workers[j] == worker)
					{
						only_remove = 1;
						break;
					}
			}
			if(only_remove ||
			   (sc_w->resize_ack.receiver_sched_ctx != -1 && sc_w->resize_ack.receiver_sched_ctx == (int)sched_ctx))
			{
				resize_ack = &sc_w->resize_ack;
				sender_sched_ctx = hypervisor.sched_ctxs[i];
				starpu_pthread_mutex_unlock(&sc_w->mutex);
				break;
			}
			starpu_pthread_mutex_unlock(&sc_w->mutex);
		}
	}

	/* if there is no ctx waiting for its ack return 1*/
	if(resize_ack == NULL)
		return 1;

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
			starpu_pthread_mutex_unlock(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
			return resize_completed;
		}
		starpu_pthread_mutex_unlock(&hypervisor.sched_ctx_w[sender_sched_ctx].mutex);
	}
	return 0;
}

/* Enqueue a resize request for 'sched_ctx', to be executed when the
 * 'task_tag' tasks of 'sched_ctx' complete.  */
void sc_hypervisor_resize(unsigned sched_ctx, int task_tag)
{
	struct resize_request_entry *entry;

	entry = malloc(sizeof *entry);
	STARPU_ASSERT(entry != NULL);

	entry->sched_ctx = sched_ctx;
	entry->task_tag = task_tag;

	starpu_pthread_mutex_lock(&hypervisor.resize_mut[sched_ctx]);
	HASH_ADD_INT(hypervisor.resize_requests[sched_ctx], task_tag, entry);
	starpu_pthread_mutex_unlock(&hypervisor.resize_mut[sched_ctx]);
}

/* notifies the hypervisor that the worker is no longer idle and a new task was pushed on its queue */
static void notify_idle_end(unsigned sched_ctx, int worker)
{
	if(hypervisor.resize[sched_ctx])
		hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker] = 0.0;

	if(hypervisor.policy.handle_idle_end)
		hypervisor.policy.handle_idle_end(sched_ctx, worker);

}

/* notifies the hypervisor that the worker spent another cycle in idle time */
static void notify_idle_cycle(unsigned sched_ctx, int worker, double idle_time)
{
	if(hypervisor.resize[sched_ctx])
	{
		struct sc_hypervisor_wrapper *sc_w = &hypervisor.sched_ctx_w[sched_ctx];
		sc_w->current_idle_time[worker] += idle_time;
		if(hypervisor.policy.handle_idle_cycle)
		{
			hypervisor.policy.handle_idle_cycle(sched_ctx, worker);
		}
	}
	return;
}

/* notifies the hypervisor that a new task was pushed on the queue of the worker */
static void notify_pushed_task(unsigned sched_ctx, int worker)
{
	hypervisor.sched_ctx_w[sched_ctx].pushed_tasks[worker]++;
	if(hypervisor.sched_ctx_w[sched_ctx].total_flops != 0.0 && hypervisor.sched_ctx_w[sched_ctx].start_time == 0.0)
		hypervisor.sched_ctx_w[sched_ctx].start_time = starpu_timing_now();

	if(hypervisor.sched_ctx_w[sched_ctx].total_flops != 0.0 && hypervisor.sched_ctx_w[sched_ctx].real_start_time == 0.0)
		hypervisor.sched_ctx_w[sched_ctx].real_start_time = starpu_timing_now();

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

/* notifies the hypervisor that a task was poped from the queue of the worker */
static void notify_poped_task(unsigned sched_ctx, int worker, struct starpu_task *task, size_t data_size, uint32_t footprint)
{
	hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker]++;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_flops[worker] += task->flops;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_data[worker] += data_size ;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_tasks[worker]++ ;
	hypervisor.sched_ctx_w[sched_ctx].total_elapsed_flops[worker] += task->flops;
	hypervisor.sched_ctx_w[sched_ctx].remaining_flops -= task->flops; //sc_hypervisor_get_elapsed_flops_per_sched_ctx(&hypervisor.sched_ctx_w[sched_ctx]);

	if(hypervisor.resize[sched_ctx])
	{	
		if(hypervisor.policy.handle_poped_task)
			hypervisor.policy.handle_poped_task(sched_ctx, worker, task, footprint);
	}
	_ack_resize_completed(sched_ctx, worker);
	if(hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker] % 200 == 0)
		_print_current_time();
}

/* notifies the hypervisor that a tagged task has just been executed */
static void notify_post_exec_hook(unsigned sched_ctx, int task_tag)
{
	STARPU_ASSERT(task_tag > 0);

	unsigned conf_sched_ctx;
	unsigned i;
	starpu_pthread_mutex_lock(&act_hypervisor_mutex);
	unsigned ns = hypervisor.nsched_ctxs;
	starpu_pthread_mutex_unlock(&act_hypervisor_mutex);

	for(i = 0; i < ns; i++)
	{
		struct configuration_entry *entry;

		conf_sched_ctx = hypervisor.sched_ctxs[i];
		starpu_pthread_mutex_lock(&hypervisor.conf_mut[conf_sched_ctx]);

		HASH_FIND_INT(hypervisor.configurations[conf_sched_ctx], &task_tag, entry);

		if (entry != NULL)
		{
			struct sc_hypervisor_policy_config *config = entry->configuration;

			sc_hypervisor_set_config(conf_sched_ctx, config);
			HASH_DEL(hypervisor.configurations[conf_sched_ctx], entry);
			free(config);
		}
		starpu_pthread_mutex_unlock(&hypervisor.conf_mut[conf_sched_ctx]);
	}

	if(hypervisor.resize[sched_ctx])
	{
		starpu_pthread_mutex_lock(&hypervisor.resize_mut[sched_ctx]);

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
		starpu_pthread_mutex_unlock(&hypervisor.resize_mut[sched_ctx]);
	}
	return;
}

static void notify_submitted_job(struct starpu_task *task, uint32_t footprint)
{
	starpu_pthread_mutex_lock(&act_hypervisor_mutex);
	hypervisor.sched_ctx_w[task->sched_ctx].submitted_flops += task->flops;
	starpu_pthread_mutex_unlock(&act_hypervisor_mutex);

	if(hypervisor.policy.handle_submitted_job && !type_of_tasks_known)
		hypervisor.policy.handle_submitted_job(task->cl, task->sched_ctx, footprint);
}

void sc_hypervisor_set_type_of_task(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint)
{
	type_of_tasks_known = 1;
	if(hypervisor.policy.handle_submitted_job)
		hypervisor.policy.handle_submitted_job(cl, sched_ctx, footprint);
}

static void notify_delete_context(unsigned sched_ctx)
{
	_print_current_time();
	sc_hypervisor_unregister_ctx(sched_ctx);
}

void sc_hypervisor_size_ctxs(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers)
{
	starpu_pthread_mutex_lock(&act_hypervisor_mutex);
	unsigned curr_nsched_ctxs = sched_ctxs == NULL ? hypervisor.nsched_ctxs : nsched_ctxs;
	int *curr_sched_ctxs = sched_ctxs == NULL ? hypervisor.sched_ctxs : sched_ctxs;
	starpu_pthread_mutex_unlock(&act_hypervisor_mutex);
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

int* sc_hypervisor_get_sched_ctxs()
{
	return hypervisor.sched_ctxs;
}

int sc_hypervisor_get_nsched_ctxs()
{
	int ns;
	ns = hypervisor.nsched_ctxs;
	return ns;
}

void sc_hypervisor_save_size_req(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers)
{
	hypervisor.sr = (struct size_request*)malloc(sizeof(struct size_request));
	hypervisor.sr->sched_ctxs = sched_ctxs;
	hypervisor.sr->nsched_ctxs = nsched_ctxs;
	hypervisor.sr->workers = workers;
	hypervisor.sr->nworkers = nworkers;
}

unsigned sc_hypervisor_get_size_req(int **sched_ctxs, int* nsched_ctxs, int **workers, int *nworkers)
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

double sc_hypervisor_get_velocity(struct sc_hypervisor_wrapper *sc_w, enum starpu_worker_archtype arch)
{

	double velocity = sc_hypervisorsc_hypervisor_get_velocity_per_worker_type(sc_w, arch);
	if(velocity == -1.0)
		velocity = sc_hypervisor_get_ref_velocity_per_worker_type(sc_w, arch);
	if(velocity == -1.0)
		velocity = arch == STARPU_CPU_WORKER ? 5.0 : 100.0;
       
	return velocity;
}
