/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  INRIA
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

#include <core/sched_policy.h>
#include <core/sched_ctx.h>
#include <common/utils.h>

extern struct worker_collection worker_list;
static pthread_mutex_t sched_ctx_manag = PTHREAD_MUTEX_INITIALIZER;
struct starpu_task stop_submission_task = STARPU_TASK_INITIALIZER;
pthread_key_t sched_ctx_key;
unsigned with_hypervisor = 0;
double max_time_worker_on_ctx = -1.0;

static unsigned _starpu_get_first_free_sched_ctx(struct _starpu_machine_config *config);
static unsigned _starpu_worker_get_first_free_sched_ctx(struct _starpu_worker *worker);

static unsigned _starpu_worker_get_sched_ctx_id(struct _starpu_worker *worker, unsigned sched_ctx_id);

static void change_worker_sched_ctx(unsigned sched_ctx_id)
{
	int workerid = starpu_worker_get_id();
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);

	int worker_sched_ctx_id = _starpu_worker_get_sched_ctx_id(worker, sched_ctx_id);
	/* if the worker is not in the ctx's list it means the update concerns the addition of ctxs*/
	if(worker_sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
	{
		worker_sched_ctx_id = _starpu_worker_get_first_free_sched_ctx(worker);
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
		/* add context to worker */
		worker->sched_ctx[worker_sched_ctx_id] = sched_ctx;
		worker->nsched_ctxs++;	
		worker->active_ctx = sched_ctx_id;
	}
	else 
	{
		/* remove context from worker */
		if(worker->sched_ctx[worker_sched_ctx_id]->sched_policy)
			worker->sched_ctx[worker_sched_ctx_id]->sched_policy->remove_workers(sched_ctx_id, &worker->workerid, 1);
		worker->sched_ctx[worker_sched_ctx_id] = NULL;
		worker->nsched_ctxs--;
		starpu_set_turn_to_other_ctx(worker->workerid, sched_ctx_id);
	}
}

static void update_workers_func(void *buffers[] __attribute__ ((unused)), void *_args)
{
	int sched_ctx_id = (int)_args;
	change_worker_sched_ctx(sched_ctx_id);
}

struct starpu_codelet sched_ctx_info_cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cuda_func = update_workers_func,
	.cpu_func = update_workers_func,
	.opencl_func = update_workers_func,
	.nbuffers = 0
};

static void _starpu_update_workers(int *workerids, int nworkers, int sched_ctx_id)
{
	int i;
	struct _starpu_worker *worker[nworkers];
 	struct _starpu_worker *curr_worker = _starpu_get_local_worker_key();

	for(i = 0; i < nworkers; i++)
	{
		worker[i] = _starpu_get_worker_struct(workerids[i]);

		/* if the current thread requires resize it's no need
		   to send itsefl a message in order to change its 
		   sched_ctx info */
		if(curr_worker && curr_worker == worker[i])
			change_worker_sched_ctx(sched_ctx_id);
		else
		{			
			worker[i]->tasks[sched_ctx_id] = starpu_task_create();
			worker[i]->tasks[sched_ctx_id]->cl = &sched_ctx_info_cl;
			worker[i]->tasks[sched_ctx_id]->cl_arg = (void*)sched_ctx_id;
			worker[i]->tasks[sched_ctx_id]->execute_on_a_specific_worker = 1;
			worker[i]->tasks[sched_ctx_id]->workerid = workerids[i];
			worker[i]->tasks[sched_ctx_id]->destroy = 1;
			worker[i]->tasks[sched_ctx_id]->control_task = 1;
			int worker_sched_ctx_id = _starpu_worker_get_sched_ctx_id(worker[i], sched_ctx_id);
                        /* if the ctx is not in the worker's list it means the update concerns the addition of ctxs*/
                        if(worker_sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
                                worker[i]->tasks[sched_ctx_id]->priority = 1;

			_starpu_exclude_task_from_dag(worker[i]->tasks[sched_ctx_id]);

			_starpu_task_submit_internally(worker[i]->tasks[sched_ctx_id]);
		}		
	}
}

void starpu_stop_task_submission(unsigned sched_ctx_id)
{
	_starpu_exclude_task_from_dag(&stop_submission_task);
	_starpu_task_submit_internally(&stop_submission_task);
}

static void _starpu_add_workers_to_sched_ctx(struct _starpu_sched_ctx *sched_ctx, int *workerids, int nworkers, 
				       int *added_workers, int *n_added_workers)
{
	struct worker_collection *workers = sched_ctx->workers;
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();

	int nworkers_to_add = nworkers == -1 ? config->topology.nworkers : nworkers;
	int workers_to_add[nworkers_to_add];

	int i = 0;
	for(i = 0; i < nworkers_to_add; i++)
	{
		/* added_workers is NULL for the call of this func at the creation of the context*/
		/* if the function is called at the creation of the context it's no need to do this verif */
		if(added_workers)
		{
			int worker = workers->add(workers, (workerids == NULL ? i : workerids[i]));
			if(worker >= 0)
			{
				added_workers[(*n_added_workers)++] = worker;		
			}
		}
		else
		{
			int worker = (workerids == NULL ? i : workerids[i]); 
			workers->add(workers, worker);
			workers_to_add[i] = worker;
		}
	}

	if(added_workers)
	{
		if(*n_added_workers > 0)
			sched_ctx->sched_policy->add_workers(sched_ctx->id, added_workers, *n_added_workers);	
	}
	else
		sched_ctx->sched_policy->add_workers(sched_ctx->id, workers_to_add, nworkers_to_add);		

	return;
}

static void _starpu_remove_workers_from_sched_ctx(struct _starpu_sched_ctx *sched_ctx, int *workerids, 
						  int nworkers, int *removed_workers, int *n_removed_workers)
{
	struct worker_collection *workers = sched_ctx->workers;

	int i = 0;

	for(i = 0; i < nworkers; i++)
	{
		if(workers->nworkers > 0)
		{
			int worker = workers->remove(workers, workerids[i]);
			if(worker >= 0)
				removed_workers[(*n_removed_workers)++] = worker;
		}
	}

	if(*n_removed_workers)
		sched_ctx->sched_policy->remove_workers(sched_ctx->id, removed_workers, *n_removed_workers);
	return;
}

struct _starpu_sched_ctx*  _starpu_create_sched_ctx(const char *policy_name, int *workerids, 
				  int nworkers_ctx, unsigned is_initial_sched,
				  const char *sched_name)
{
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();

	_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx_manag);
	STARPU_ASSERT(config->topology.nsched_ctxs < STARPU_NMAX_SCHED_CTXS);

	unsigned id = _starpu_get_first_free_sched_ctx(config);

	struct _starpu_sched_ctx *sched_ctx = &config->sched_ctxs[id];
	sched_ctx->id = id;

	config->topology.nsched_ctxs++;	
	_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx_manag);

	int nworkers = config->topology.nworkers;
	
	STARPU_ASSERT(nworkers_ctx <= nworkers);
  
	_STARPU_PTHREAD_MUTEX_INIT(&sched_ctx->changing_ctx_mutex, NULL);
	_STARPU_PTHREAD_MUTEX_INIT(&sched_ctx->empty_ctx_mutex, NULL);

	starpu_task_list_init(&sched_ctx->empty_ctx_tasks);

	sched_ctx->sched_policy = (struct starpu_sched_policy*)malloc(sizeof(struct starpu_sched_policy));
	sched_ctx->is_initial_sched = is_initial_sched;
	sched_ctx->name = sched_name;

	_starpu_barrier_counter_init(&sched_ctx->tasks_barrier, 0);

	/* initialise all sync structures bc the number of workers can modify */
	sched_ctx->sched_mutex = (pthread_mutex_t**)malloc(STARPU_NMAXWORKERS * sizeof(pthread_mutex_t*));
	sched_ctx->sched_cond = (pthread_cond_t**)malloc(STARPU_NMAXWORKERS * sizeof(pthread_cond_t*));

	
	/*init the strategy structs and the worker_collection of the ressources of the context */
	_starpu_init_sched_policy(config, sched_ctx, policy_name);

	/* construct the collection of workers(list/tree/etc.) */
	sched_ctx->workers->workerids = sched_ctx->workers->init(sched_ctx->workers);
	sched_ctx->workers->nworkers = 0;

	/* after having an worker_collection on the ressources add them */
	_starpu_add_workers_to_sched_ctx(sched_ctx, workerids, nworkers_ctx, NULL, NULL);


	/* if we create the initial big sched ctx we can update workers' status here
	   because they haven't been launched yet */
	if(is_initial_sched)
	{
		int i;
		for(i = 0; i < nworkers; i++)
		{
			struct _starpu_worker *worker = _starpu_get_worker_struct(i);
			worker->sched_ctx[_starpu_worker_get_first_free_sched_ctx(worker)] = sched_ctx;
			worker->nsched_ctxs++;
		}
	}

	int w;
	for(w = 0; w < STARPU_NMAXWORKERS; w++)
	{
		sched_ctx->pop_counter[w] = 0;
	}
	
	return sched_ctx;
}

static void _get_workers(int min, int max, int *workers, int *nw, enum starpu_archtype arch, unsigned allow_overlap)
{
	int pus[max];
	int npus = 0; 
	int i;
	int n = 0;
		
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	if(config->topology.nsched_ctxs == 1)
	{
		/*we have all available resources */
		npus = starpu_worker_get_nids_by_type(arch, pus, max);
/*TODO: hierarchical ctxs: get max good workers: close one to another */
		for(i = 0; i < npus; i++)
			workers[(*nw)++] = pus[i];
	}
	else
	{
		unsigned enough_ressources = 0;
		npus = starpu_worker_get_available_ids_by_type(arch, pus, max);
       
		for(i = 0; i < npus; i++)
			workers[(*nw)++] = pus[i];
		
		if(npus == max)
			/*we have enough available resources */
			enough_ressources = 1;

		if(!enough_ressources && npus >= min)
			/*we have enough available resources */
			enough_ressources = 1;

		if(!enough_ressources)
		{
			/* try to get ressources from ctx who have more than the min of workers they need */
			int s;
                        for(s = 1; s < STARPU_NMAX_SCHED_CTXS; s++)
                        {
				if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
				{
					int _npus = 0;
					int _pus[STARPU_NMAXWORKERS];
					_npus = starpu_get_workers_of_sched_ctx(config->sched_ctxs[s].id, _pus, arch);
					int ctx_min = arch == STARPU_CPU_WORKER ? config->sched_ctxs[s].min_ncpus : config->sched_ctxs[s].min_ngpus;
					if(_npus > ctx_min)
					{
                                                if(npus < min)
						{
							n = (_npus - ctx_min) > (min - npus) ? min - npus : (_npus - ctx_min);
							npus += n;
						}
/*TODO: hierarchical ctxs: get n good workers: close to the other ones I already assigned to the ctx */
						for(i = 0; i < n; i++)
							workers[(*nw)++] = _pus[i];
						starpu_remove_workers_from_sched_ctx(_pus, n, config->sched_ctxs[s].id);
					}

				}
			}
			if(npus >= min)
				enough_ressources = 1;

		}
		
		if(!enough_ressources)
		{
			/* if there is no available workers to satisfy the  minimum required 
			 give them workers proportional to their requirements*/
			int global_npus = starpu_worker_get_count_by_type(arch);
			
			int req_npus = 0;

			int s;
			for(s = 1; s < STARPU_NMAX_SCHED_CTXS; s++)
				if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
					req_npus += arch == STARPU_CPU_WORKER ? config->sched_ctxs[s].min_ncpus : config->sched_ctxs[s].min_ngpus;

			req_npus += min;
			
			for(s = 1; s < STARPU_NMAX_SCHED_CTXS; s++)
			{
				if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
				{
					int ctx_min = arch == STARPU_CPU_WORKER ? config->sched_ctxs[s].min_ncpus : config->sched_ctxs[s].min_ngpus;
					double needed_npus = ((double)ctx_min * (double)global_npus) / (double)req_npus;

					int _npus = 0;
					int _pus[STARPU_NMAXWORKERS];
				
					_npus = starpu_get_workers_of_sched_ctx(config->sched_ctxs[s].id, _pus, arch);					
					if(needed_npus < (double)_npus)
					{
						double npus_to_rem = (double)_npus - needed_npus;
						int x = floor(npus_to_rem);
						double x_double = (double)x;
						double diff = npus_to_rem - x_double;
						int npus_to_remove = diff >= 0.5 ? x+1 : x;

						int pus_to_remove[npus_to_remove];
						int c = 0;
						
/*TODO: hierarchical ctxs: get npus_to_remove good workers: close to the other ones I already assigned to the ctx */
						for(i = _npus-1; i >= (_npus - npus_to_remove); i--)
						{
							workers[(*nw)++] = _pus[i];
							pus_to_remove[c++] = _pus[i];
						}
						if(!allow_overlap)
							starpu_remove_workers_from_sched_ctx(pus_to_remove, npus_to_remove, config->sched_ctxs[s].id);
					}

				}
			}
		}
	}
}

unsigned starpu_create_sched_ctx_inside_interval(const char *policy_name, const char *sched_name, 
						 int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus,
						 unsigned allow_overlap)
{
	struct _starpu_sched_ctx *sched_ctx = NULL;
	int workers[max_ncpus + max_ngpus];
	int nw = 0;
	_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx_manag);
	_get_workers(min_ncpus, max_ncpus, workers, &nw, STARPU_CPU_WORKER, allow_overlap);
	_get_workers(min_ngpus, max_ngpus, workers, &nw, STARPU_CUDA_WORKER, allow_overlap);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx_manag);
	int i;
	printf("%d: ", nw);
	for(i = 0; i < nw; i++)
		printf("%d ", workers[i]);
	printf("\n");
	sched_ctx = _starpu_create_sched_ctx(policy_name, workers, nw, 0, sched_name);
	sched_ctx->min_ncpus = min_ncpus;
	sched_ctx->max_ncpus = max_ncpus;
	sched_ctx->min_ngpus = min_ngpus;
	sched_ctx->max_ngpus = max_ngpus;
	
	_starpu_update_workers(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, sched_ctx->id);
#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	sched_ctx->perf_counters = NULL;
#endif
	return sched_ctx->id;
	
}
unsigned starpu_create_sched_ctx(const char *policy_name, int *workerids, 
				 int nworkers, const char *sched_name)
{
	struct _starpu_sched_ctx *sched_ctx = NULL;
	sched_ctx = _starpu_create_sched_ctx(policy_name, workerids, nworkers, 0, sched_name);

	_starpu_update_workers(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, sched_ctx->id);
#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	sched_ctx->perf_counters = NULL;
#endif
	return sched_ctx->id;
}

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
void starpu_set_perf_counters(unsigned sched_ctx_id, struct starpu_performance_counters *perf_counters)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->perf_counters = perf_counters;
	return;
}
#endif

/* free all structures for the context */
static void _starpu_delete_sched_ctx(struct _starpu_sched_ctx *sched_ctx)
{
	_starpu_deinit_sched_policy(sched_ctx);		
	free(sched_ctx->sched_policy);
	free(sched_ctx->sched_mutex);
	free(sched_ctx->sched_cond);

	sched_ctx->sched_policy = NULL;
	sched_ctx->sched_mutex = NULL;
	sched_ctx->sched_cond = NULL;

	_STARPU_PTHREAD_MUTEX_DESTROY(&sched_ctx->changing_ctx_mutex);
	_STARPU_PTHREAD_MUTEX_DESTROY(&sched_ctx->empty_ctx_mutex);

	struct _starpu_machine_config *config = _starpu_get_machine_config();
	_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx_manag);
	config->topology.nsched_ctxs--;
	sched_ctx->id = STARPU_NMAX_SCHED_CTXS;
	_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx_manag);
}

void starpu_delete_sched_ctx(unsigned sched_ctx_id, unsigned inheritor_sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct _starpu_sched_ctx *inheritor_sched_ctx = _starpu_get_sched_ctx_struct(inheritor_sched_ctx_id);

	_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->changing_ctx_mutex);
	_starpu_update_workers(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, sched_ctx->id);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->changing_ctx_mutex);

	/*if both of them have all the ressources is pointless*/
	/*trying to transfer ressources from one ctx to the other*/
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	int nworkers = config->topology.nworkers;

	if(!(sched_ctx->workers->nworkers == nworkers && sched_ctx->workers->nworkers == inheritor_sched_ctx->workers->nworkers) && sched_ctx->workers->nworkers > 0 && inheritor_sched_ctx_id != STARPU_NMAX_SCHED_CTXS)
	{
		starpu_add_workers_to_sched_ctx(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, inheritor_sched_ctx_id);
	}

	if(!_starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx_id) && !_starpu_wait_for_all_tasks_of_sched_ctx(0))
		_starpu_delete_sched_ctx(sched_ctx);
	return;	
}

/* called after the workers are terminated so we don't have anything else to do but free the memory*/
void _starpu_delete_all_sched_ctxs()
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(i);
		if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
		{
			_starpu_barrier_counter_destroy(&sched_ctx->tasks_barrier);
			_starpu_delete_sched_ctx(sched_ctx);
		}
	}
	return;
}

static void _starpu_check_workers(int *workerids, int nworkers)
{
        struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
        int nworkers_conf = config->topology.nworkers;

	int i;
	for(i = 0; i < nworkers; i++)
	{
		/* take care the user does not ask for a resource that does not exist */
		STARPU_ASSERT(workerids[i] >= 0 &&  workerids[i] <= nworkers_conf);
	}		
}

void _starpu_fetch_tasks_from_empty_ctx_list(struct _starpu_sched_ctx *sched_ctx)
{
	unsigned unlocked = 0;
	_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->empty_ctx_mutex);
	while(!starpu_task_list_empty(&sched_ctx->empty_ctx_tasks))
	{
		if(unlocked)
			_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->empty_ctx_mutex);
		struct starpu_task *old_task = starpu_task_list_pop_back(&sched_ctx->empty_ctx_tasks);
		unlocked = 1;
		_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->empty_ctx_mutex);
		
		if(old_task == &stop_submission_task)
			break;

		struct _starpu_job *old_j = _starpu_get_job_associated_to_task(old_task);
		int ret = _starpu_push_task(old_j);
		/* if we should stop poping from empty ctx tasks */
		if(ret == -1) break;
	}
	if(!unlocked)
		_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->empty_ctx_mutex);
	return;

}
void starpu_add_workers_to_sched_ctx(int *workers_to_add, int nworkers_to_add, unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int added_workers[nworkers_to_add];
	int n_added_workers = 0;

	STARPU_ASSERT(workers_to_add != NULL && nworkers_to_add > 0);
	_starpu_check_workers(workers_to_add, nworkers_to_add);

	_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->changing_ctx_mutex);
	_starpu_add_workers_to_sched_ctx(sched_ctx, workers_to_add, nworkers_to_add, added_workers, &n_added_workers);

	if(n_added_workers > 0)
	{
		_starpu_update_workers(added_workers, n_added_workers, sched_ctx->id);
	}

	_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->changing_ctx_mutex);
	
	_starpu_fetch_tasks_from_empty_ctx_list(sched_ctx);

	return;
}

void starpu_remove_workers_from_sched_ctx(int *workers_to_remove, int nworkers_to_remove, unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int removed_workers[sched_ctx->workers->nworkers];
	int n_removed_workers = 0;

	_starpu_check_workers(workers_to_remove, nworkers_to_remove);

	_STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->changing_ctx_mutex);
	_starpu_remove_workers_from_sched_ctx(sched_ctx, workers_to_remove, nworkers_to_remove, removed_workers, &n_removed_workers);

	if(n_removed_workers > 0)
		_starpu_update_workers(removed_workers, n_removed_workers, sched_ctx->id);

	_STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->changing_ctx_mutex);	       
	return;
}

/* unused sched_ctx have the id STARPU_NMAX_SCHED_CTXS */
void _starpu_init_all_sched_ctxs(struct _starpu_machine_config *config)
{
	pthread_key_create(&sched_ctx_key, NULL);

	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		config->sched_ctxs[i].id = STARPU_NMAX_SCHED_CTXS;

	char* max_time_on_ctx = getenv("STARPU_MAX_TIME_ON_CTX");
	if (max_time_on_ctx != NULL)
		max_time_worker_on_ctx = atof(max_time_on_ctx);

	return;
}

/* unused sched_ctx pointers of a worker are NULL */
void _starpu_init_sched_ctx_for_worker(unsigned workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	worker->sched_ctx = (struct _starpu_sched_ctx**)malloc(STARPU_NMAX_SCHED_CTXS * sizeof(struct _starpu_sched_ctx*));
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		worker->sched_ctx[i] = NULL;

	return;
}

/* sched_ctx aren't necessarly one next to another */
/* for eg when we remove one its place is free */
/* when we add  new one we reuse its place */
static unsigned _starpu_get_first_free_sched_ctx(struct _starpu_machine_config *config)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(config->sched_ctxs[i].id == STARPU_NMAX_SCHED_CTXS)
			return i;

	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

static unsigned _starpu_worker_get_first_free_sched_ctx(struct _starpu_worker *worker)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(worker->sched_ctx[i] == NULL)
			return i;
	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

static unsigned _starpu_worker_get_sched_ctx_id(struct _starpu_worker *worker, unsigned sched_ctx_id)
{
	unsigned to_be_deleted = STARPU_NMAX_SCHED_CTXS;
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(worker->sched_ctx[i] != NULL)
			if(worker->sched_ctx[i]->id == sched_ctx_id)
				return i;
			else if(worker->sched_ctx[i]->id == STARPU_NMAX_SCHED_CTXS)
				to_be_deleted = i;

	/* little bit of a hack be carefull */
	if(to_be_deleted != STARPU_NMAX_SCHED_CTXS)
		return to_be_deleted;
	return STARPU_NMAX_SCHED_CTXS;
}

int _starpu_wait_for_all_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
	  return -EDEADLK;
	
	return _starpu_barrier_counter_wait_for_empty_counter(&sched_ctx->tasks_barrier);
}

void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_starpu_barrier_counter_decrement_until_empty_counter(&sched_ctx->tasks_barrier);
}

void _starpu_increment_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_starpu_barrier_counter_increment(&sched_ctx->tasks_barrier);
}

void starpu_set_sched_ctx(unsigned *sched_ctx)
{
	pthread_setspecific(sched_ctx_key, (void*)sched_ctx);
}

unsigned starpu_get_sched_ctx()
{
	unsigned *sched_ctx = (unsigned*)pthread_getspecific(sched_ctx_key);
	if(sched_ctx == NULL)
		return STARPU_NMAX_SCHED_CTXS;
	STARPU_ASSERT(*sched_ctx < STARPU_NMAX_SCHED_CTXS);
	return *sched_ctx;
}

void starpu_notify_hypervisor_exists()
{
	with_hypervisor = 1;
}

unsigned starpu_check_if_hypervisor_exists()
{
	return with_hypervisor;
}

unsigned _starpu_get_nsched_ctxs()
{
	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	return config->topology.nsched_ctxs;
}

void starpu_set_sched_ctx_policy_data(unsigned sched_ctx_id, void* policy_data)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->policy_data = policy_data;
}

void* starpu_get_sched_ctx_policy_data(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->policy_data;
}

pthread_mutex_t *_starpu_get_sched_mutex(struct _starpu_sched_ctx *sched_ctx, int workerid)
{
        if(sched_ctx->sched_mutex)
                return sched_ctx->sched_mutex[workerid];
	else
                return NULL;
}

void starpu_worker_set_sched_condition(unsigned sched_ctx_id, int workerid, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if(sched_ctx->sched_mutex && sched_ctx->sched_cond)
	{
		sched_ctx->sched_mutex[workerid] = sched_mutex;
		sched_ctx->sched_cond[workerid] = sched_cond;
	}
}

void starpu_worker_get_sched_condition(unsigned sched_ctx_id, int workerid, pthread_mutex_t **sched_mutex, pthread_cond_t **sched_cond)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	*sched_mutex = sched_ctx->sched_mutex[workerid];
	*sched_cond = sched_ctx->sched_cond[workerid];

	/* the tasks concerning changings of the the ctxs were not executed in order */
	if(!*sched_mutex)
	{
		struct _starpu_worker *workerarg = _starpu_get_worker_struct(workerid);
		*sched_mutex = &workerarg->sched_mutex;
		*sched_cond = &workerarg->sched_cond;
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, *sched_mutex, *sched_cond);
	}

}

void starpu_worker_init_sched_condition(unsigned sched_ctx_id, int workerid)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->sched_mutex[workerid] = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	sched_ctx->sched_cond[workerid] = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
	_STARPU_PTHREAD_MUTEX_INIT(sched_ctx->sched_mutex[workerid], NULL);
	_STARPU_PTHREAD_COND_INIT(sched_ctx->sched_cond[workerid], NULL);
}

void starpu_worker_deinit_sched_condition(unsigned sched_ctx_id, int workerid)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_STARPU_PTHREAD_MUTEX_DESTROY(sched_ctx->sched_mutex[workerid]);
	_STARPU_PTHREAD_COND_DESTROY(sched_ctx->sched_cond[workerid]);
	free(sched_ctx->sched_mutex[workerid]);
	free(sched_ctx->sched_cond[workerid]);
}

struct worker_collection* starpu_create_worker_collection_for_sched_ctx(unsigned sched_ctx_id, int worker_collection_type)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->workers = (struct worker_collection*)malloc(sizeof(struct worker_collection));

	switch(worker_collection_type)
	{
	case WORKER_LIST:
		sched_ctx->workers->has_next = worker_list.has_next;
		sched_ctx->workers->get_next = worker_list.get_next;
		sched_ctx->workers->add = worker_list.add;
		sched_ctx->workers->remove = worker_list.remove;
		sched_ctx->workers->init = worker_list.init;
		sched_ctx->workers->deinit = worker_list.deinit;
		sched_ctx->workers->init_cursor = worker_list.init_cursor;
		sched_ctx->workers->deinit_cursor = worker_list.deinit_cursor;
		sched_ctx->workers->type = WORKER_LIST; 
		break;
	}

	return sched_ctx->workers;
}

void starpu_delete_worker_collection_for_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->workers->deinit(sched_ctx->workers);

	free(sched_ctx->workers);
}

struct worker_collection* starpu_get_worker_collection_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->workers;
}

int starpu_get_workers_of_sched_ctx(unsigned sched_ctx_id, int *pus, enum starpu_archtype arch)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	struct worker_collection *workers = sched_ctx->workers;
        int worker;

	int npus = 0;

        if(workers->init_cursor)
                workers->init_cursor(workers);

        while(workers->has_next(workers))
        {
                worker = workers->get_next(workers);
		enum starpu_archtype curr_arch = starpu_worker_get_type(worker);
		if(curr_arch == arch)
			pus[npus++] = worker;
	}

        if(workers->init_cursor)
                workers->deinit_cursor(workers);
	return npus;
}

pthread_mutex_t* starpu_get_changing_ctx_mutex(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return &sched_ctx->changing_ctx_mutex;
}

unsigned starpu_get_nworkers_of_sched_ctx(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if(sched_ctx != NULL)
		return sched_ctx->workers->nworkers;
	else 
		return 0;

}

unsigned starpu_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2)
{
        struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
        struct _starpu_sched_ctx *sched_ctx2 = _starpu_get_sched_ctx_struct(sched_ctx_id2);

        struct worker_collection *workers = sched_ctx->workers;
        struct worker_collection *workers2 = sched_ctx2->workers;
        int worker, worker2;
        int shared_workers = 0;

        if(workers->init_cursor)
                workers->init_cursor(workers);

        if(workers2->init_cursor)
                workers2->init_cursor(workers2);

        while(workers->has_next(workers))
        {
                worker = workers->get_next(workers);
                while(workers2->has_next(workers2))
		{
                        worker2 = workers2->get_next(workers2);
                        if(worker == worker2)
				shared_workers++;
                }
        }

        if(workers->init_cursor)
                workers->deinit_cursor(workers);

        if(workers2->init_cursor)
                workers2->deinit_cursor(workers2);

	return shared_workers;
}

unsigned starpu_worker_belongs_to_sched_ctx(int workerid, unsigned sched_ctx_id)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		if(worker->sched_ctx[i] && worker->sched_ctx[i]->id == sched_ctx_id)
			return 1;
	}
	return 0;
}

unsigned starpu_are_overlapping_ctxs_on_worker(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	return worker->nsched_ctxs > 1;
}

unsigned starpu_is_ctxs_turn(int workerid, unsigned sched_ctx_id)
{
	if(max_time_worker_on_ctx == -1.0) return 1;

	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	return worker->active_ctx == sched_ctx_id;
}

void starpu_set_turn_to_other_ctx(int workerid, unsigned sched_ctx_id)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);

	struct _starpu_sched_ctx *other_sched_ctx = NULL;
	struct _starpu_sched_ctx *active_sched_ctx = NULL;
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		other_sched_ctx = worker->sched_ctx[i];
		if(other_sched_ctx != NULL && other_sched_ctx->id != STARPU_NMAX_SCHED_CTXS && 
		   other_sched_ctx->id != 0 && other_sched_ctx->id != sched_ctx_id)
		{
			worker->active_ctx = other_sched_ctx->id;
			active_sched_ctx = other_sched_ctx;
			break;
		}
	}		

	if(worker->active_ctx != sched_ctx_id)
	{
		_starpu_fetch_tasks_from_empty_ctx_list(active_sched_ctx);
	}
}

double starpu_get_max_time_worker_on_ctx(void)
{
	return max_time_worker_on_ctx;	
}

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR

void starpu_call_poped_task_cb(int workerid, unsigned sched_ctx_id, double flops)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if(sched_ctx != NULL && sched_ctx_id != 0 && sched_ctx_id != STARPU_NMAX_SCHED_CTXS
	   && sched_ctx->perf_counters != NULL)
		sched_ctx->perf_counters->notify_poped_task(sched_ctx_id, workerid, flops);
}

void starpu_call_pushed_task_cb(int workerid, unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	if(sched_ctx != NULL && sched_ctx_id != 0 && sched_ctx_id != STARPU_NMAX_SCHED_CTXS
	   && sched_ctx->perf_counters != NULL)
		sched_ctx->perf_counters->notify_pushed_task(sched_ctx_id, workerid);
}

#endif //STARPU_USE_SCHED_CTX_HYPERVISOR
