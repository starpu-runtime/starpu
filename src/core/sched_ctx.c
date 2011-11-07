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

pthread_key_t sched_ctx_key;

struct sched_ctx_info {
	unsigned sched_ctx_id;
	struct starpu_sched_ctx *sched_ctx;
	struct starpu_worker_s *worker;
};

static unsigned _starpu_get_first_free_sched_ctx(struct starpu_machine_config_s *config);
static unsigned _starpu_worker_get_first_free_sched_ctx(struct starpu_worker_s *worker);

static unsigned _starpu_worker_get_sched_ctx_id(struct starpu_worker_s *worker, unsigned sched_ctx_id);

static void change_worker_sched_ctx( struct starpu_worker_s *worker, struct starpu_sched_ctx *sched_ctx, unsigned sched_ctx_id)
{
	if(sched_ctx != NULL)
	{
		/* add context to worker */
		worker->sched_ctx[sched_ctx_id] = sched_ctx;
		worker->nsched_ctxs++;
	}
	else
	{
		/* remove context from worker */
		worker->sched_ctx[sched_ctx_id]->sched_mutex[worker->workerid] = NULL;
		worker->sched_ctx[sched_ctx_id]->sched_cond[worker->workerid] = NULL;
		worker->sched_ctx[sched_ctx_id] = NULL;
		worker->nsched_ctxs--;
	}

}

static void update_workers_func(void *buffers[] __attribute__ ((unused)), void *_args)
{
	struct sched_ctx_info *sched_ctx_info_args = (struct sched_ctx_info*)_args;
	struct starpu_worker_s *worker = sched_ctx_info_args->worker;
	struct starpu_sched_ctx *current_sched_ctx = sched_ctx_info_args->sched_ctx;
	unsigned sched_ctx_id = sched_ctx_info_args->sched_ctx_id;
	change_worker_sched_ctx(worker, current_sched_ctx, sched_ctx_id);
}

struct starpu_codelet_t sched_ctx_info_cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cuda_func = update_workers_func,
	.cpu_func = update_workers_func,
		.opencl_func = update_workers_func,
	.nbuffers = 0
};

static void _starpu_update_workers(int *workerids, int nworkers, 
				   int sched_ctx_id, struct starpu_sched_ctx *sched_ctx)
{
	struct starpu_task *tasks[nworkers];

	int i, ret;
	struct starpu_worker_s *worker[nworkers];
	struct sched_ctx_info sched_info_args[nworkers];
 	struct starpu_worker_s *curr_worker = _starpu_get_local_worker_key();

	int worker_sched_ctx_id = -1;
	for(i = 0; i < nworkers; i++)
	{
		worker[i] = _starpu_get_worker_struct(workerids[i]);
		worker_sched_ctx_id = sched_ctx_id == -1  ? 
			_starpu_worker_get_first_free_sched_ctx(worker[i]) : 
			_starpu_worker_get_sched_ctx_id(worker[i], sched_ctx_id);

		/* if the current thread requires resize it's no need
		   to send itsefl a message in order to change its 
		   sched_ctx info */
		if(curr_worker && curr_worker == worker[i])
		{
			change_worker_sched_ctx(curr_worker, sched_ctx, worker_sched_ctx_id);
			tasks[i] = NULL;
		}
		else
		{
			
			sched_info_args[i].sched_ctx_id = worker_sched_ctx_id;
			
			sched_info_args[i].sched_ctx = sched_ctx;
			sched_info_args[i].worker = worker[i];
			
			tasks[i] = starpu_task_create();
			tasks[i]->cl = &sched_ctx_info_cl;
			tasks[i]->cl_arg = &sched_info_args[i];
			tasks[i]->execute_on_a_specific_worker = 1;
			tasks[i]->workerid = workerids[i];
			tasks[i]->detach = 0;
			tasks[i]->destroy = 0;
			
			_starpu_exclude_task_from_dag(tasks[i]);
			
			ret = _starpu_task_submit_internal(tasks[i]);
			if (ret == -ENODEV)
			{
				/* if the worker is not able to execute this tasks, we
				 * don't insist as this means the worker is not
				 * designated by the "where" bitmap */
				starpu_task_destroy(tasks[i]);
				tasks[i] = NULL;
			}
			
		}		
	}

	for (i = 0; i < nworkers; i++)
	{
		if (tasks[i])
		{
			ret = starpu_task_wait(tasks[i]);
			STARPU_ASSERT(!ret);
			starpu_task_destroy(tasks[i]);
		}
	}
}


static void _starpu_add_workers_to_sched_ctx(struct starpu_sched_ctx *sched_ctx, int *workerids, int nworkers, 
				       int *added_workers, int *n_added_workers)
{
	struct worker_collection *workers = sched_ctx->workers;
	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	
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
				added_workers[(*n_added_workers)++] = worker;		
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

static void _starpu_remove_workers_from_sched_ctx(struct starpu_sched_ctx *sched_ctx, int *workerids, unsigned nworkers, 
					    int *removed_workers, int *n_removed_workers)
{
	struct worker_collection *workers = sched_ctx->workers;

	int i = 0;
	for(i = 0; i < nworkers; i++)
	{
		int worker = workers->remove(workers, workerids[i]);
		if(worker >= 0)
			removed_workers[(*n_removed_workers)++] = worker;
	}
					   
	return;
}

struct starpu_sched_ctx*  _starpu_create_sched_ctx(const char *policy_name, int *workerids, 
				  int nworkers_ctx, unsigned is_initial_sched,
				  const char *sched_name)
{
	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	STARPU_ASSERT(config->topology.nsched_ctxs < STARPU_NMAX_SCHED_CTXS - 1);

	unsigned id = _starpu_get_first_free_sched_ctx(config);

	struct starpu_sched_ctx *sched_ctx = &config->sched_ctxs[id];
	sched_ctx->id = id;
	int nworkers = config->topology.nworkers;
	
	STARPU_ASSERT(nworkers_ctx <= nworkers);
  
	PTHREAD_MUTEX_INIT(&sched_ctx->changing_ctx_mutex, NULL);

	sched_ctx->sched_policy = malloc(sizeof(struct starpu_sched_policy_s));
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

	config->topology.nsched_ctxs++;	

	/* if we create the initial big sched ctx we can update workers' status here
	   because they haven't been launched yet */
	if(is_initial_sched)
	{
		int i;
		for(i = 0; i < nworkers; i++)
		{
			struct starpu_worker_s *worker = _starpu_get_worker_struct(i);
			worker->sched_ctx[_starpu_worker_get_first_free_sched_ctx(worker)] = sched_ctx;
			worker->nsched_ctxs++;
		}
	}
	
	return sched_ctx;
}

unsigned starpu_create_sched_ctx(const char *policy_name, int *workerids, 
			    int nworkers_ctx, const char *sched_name)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_create_sched_ctx(policy_name, workerids, nworkers_ctx, 0, sched_name);

	_starpu_update_workers(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, -1, sched_ctx);
	return sched_ctx->id;
}

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
unsigned starpu_create_sched_ctx_with_criteria(const char *policy_name, int *workerids, 
				 int nworkers_ctx, const char *sched_name,
				 struct starpu_sched_ctx_hypervisor_criteria *criteria)
{
	unsigned sched_ctx_id = starpu_create_sched_ctx(policy_name, workerids, nworkers_ctx, sched_name);
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->criteria = criteria;
	return sched_ctx_id;
}
#endif


/* free all structures for the context */
static void free_sched_ctx_mem(struct starpu_sched_ctx *sched_ctx)
{
	/* just for debug in order to seg fault if we use these structures after del */
	sched_ctx->sched_policy = NULL;
	sched_ctx->sched_mutex = NULL;
	sched_ctx->sched_cond = NULL;
	sched_ctx->workers->deinit(sched_ctx->workers);

	free(sched_ctx->workers);
	free(sched_ctx->sched_policy);
	free(sched_ctx->sched_mutex);
	free(sched_ctx->sched_cond);
	struct starpu_machine_config_s *config = _starpu_get_machine_config();
	config->topology.nsched_ctxs--;
	sched_ctx->id = STARPU_NMAX_SCHED_CTXS;

}

void starpu_delete_sched_ctx(unsigned sched_ctx_id, unsigned inheritor_sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	struct starpu_sched_ctx *inheritor_sched_ctx = _starpu_get_sched_ctx_struct(inheritor_sched_ctx_id);

	PTHREAD_MUTEX_LOCK(&sched_ctx->changing_ctx_mutex);
	_starpu_update_workers(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, sched_ctx->id, NULL);
	PTHREAD_MUTEX_UNLOCK(&sched_ctx->changing_ctx_mutex);

	/*if both of them have all the ressources is pointless*/
	/*trying to transfer ressources from one ctx to the other*/
	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	int nworkers = config->topology.nworkers;

	if(!(sched_ctx->workers->nworkers == nworkers && sched_ctx->workers->nworkers == inheritor_sched_ctx->workers->nworkers))
		starpu_add_workers_to_sched_ctx(sched_ctx->workers->workerids, sched_ctx->workers->nworkers, inheritor_sched_ctx_id);
	
	if(!starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx_id))
	{
		free_sched_ctx_mem(sched_ctx);
	}

	return;	
}

/* called after the workers are terminated so we don't have anything else to do but free the memory*/
void _starpu_delete_all_sched_ctxs()
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(i);
		if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
		{
			_starpu_deinit_sched_policy(sched_ctx);		
			_starpu_barrier_counter_destroy(&sched_ctx->tasks_barrier);
			if(sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
				free_sched_ctx_mem(sched_ctx);
		}
	}
	return;
}

static void _starpu_check_workers(int *workerids, int nworkers)
{
        struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
        int nworkers_conf = config->topology.nworkers;

	int i;
	for(i = 0; i < nworkers; i++)
	{
		/* take care the user does not ask for a resource that does not exist */
		STARPU_ASSERT(workerids[i] >= 0 &&  workerids[i] <= nworkers_conf);
	}		

}

void starpu_add_workers_to_sched_ctx(int *workers_to_add, int nworkers_to_add,
				     unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int added_workers[nworkers_to_add];
	int n_added_workers = 0;

	STARPU_ASSERT(workers_to_add != NULL && nworkers_to_add > 0);
	_starpu_check_workers(workers_to_add, nworkers_to_add);

	PTHREAD_MUTEX_LOCK(&sched_ctx->changing_ctx_mutex);
	_starpu_add_workers_to_sched_ctx(sched_ctx, workers_to_add, nworkers_to_add, added_workers, &n_added_workers);
	PTHREAD_MUTEX_UNLOCK(&sched_ctx->changing_ctx_mutex);

	if(n_added_workers > 0)
		_starpu_update_workers(added_workers, n_added_workers, -1, sched_ctx);
       
	return;
}

void starpu_remove_workers_from_sched_ctx(int *workers_to_remove, int nworkers_to_remove, 
					  unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	int removed_workers[nworkers_to_remove];
	int n_removed_workers = 0;

	STARPU_ASSERT(workers_to_remove != NULL && nworkers_to_remove > 0);
	_starpu_check_workers(workers_to_remove, nworkers_to_remove);

	PTHREAD_MUTEX_LOCK(&sched_ctx->changing_ctx_mutex);
	_starpu_remove_workers_from_sched_ctx(sched_ctx, workers_to_remove, nworkers_to_remove, removed_workers, &n_removed_workers);
	PTHREAD_MUTEX_UNLOCK(&sched_ctx->changing_ctx_mutex);	
	if(n_removed_workers > 0)
	{
		_starpu_update_workers(removed_workers, n_removed_workers, sched_ctx->id, NULL);
		sched_ctx->sched_policy->remove_workers(sched_ctx_id, removed_workers, n_removed_workers);
	} 

       
	return;
}

/* unused sched_ctx have the id STARPU_NMAX_SCHED_CTXS */
void _starpu_init_all_sched_ctxs(struct starpu_machine_config_s *config)
{
	pthread_key_create(&sched_ctx_key, NULL);

	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		config->sched_ctxs[i].id = STARPU_NMAX_SCHED_CTXS;

	return;
}

/* unused sched_ctx pointers of a worker are NULL */
void _starpu_init_sched_ctx_for_worker(unsigned workerid)
{
	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);
	worker->sched_ctx = (struct starpu_sched_ctx**)malloc(STARPU_NMAX_SCHED_CTXS * sizeof(struct starpu_sched_ctx*));
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		worker->sched_ctx[i] = NULL;

	return;
}

/* sched_ctx aren't necessarly one next to another */
/* for eg when we remove one its place is free */
/* when we add  new one we reuse its place */
static unsigned _starpu_get_first_free_sched_ctx(struct starpu_machine_config_s *config)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(config->sched_ctxs[i].id == STARPU_NMAX_SCHED_CTXS)
			return i;

	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

static unsigned _starpu_worker_get_first_free_sched_ctx(struct starpu_worker_s *worker)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(worker->sched_ctx[i] == NULL)
			return i;
	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

static unsigned _starpu_worker_get_sched_ctx_id(struct starpu_worker_s *worker, unsigned sched_ctx_id)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(worker->sched_ctx[i] != NULL && worker->sched_ctx[i]->id == sched_ctx_id)
			return i;
	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

int starpu_wait_for_all_tasks_of_worker(int workerid)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);
	
	_starpu_barrier_counter_wait_for_empty_counter(&worker->tasks_barrier);
	
	return 0;
}

int starpu_wait_for_all_tasks_of_workers(int *workerids, int nworkers_ctx){
	int ret_val = 0;
	
	struct starpu_machine_config_s *config = _starpu_get_machine_config();
	int nworkers = nworkers_ctx == -1 ? (int)config->topology.nworkers : nworkers_ctx;
	
	int workerid = -1;
	int i, n;
	
	for(i = 0; i < nworkers; i++)
	  {
		workerid = workerids == NULL ? i : workerids[i];
		n = starpu_wait_for_all_tasks_of_worker(workerid);
		ret_val = (ret_val && n);
	  }
	
	return ret_val;
}

void _starpu_decrement_nsubmitted_tasks_of_worker(int workerid)
{
	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);
	
	_starpu_barrier_counter_decrement_until_empty_counter(&worker->tasks_barrier);

	return;
}

void _starpu_increment_nsubmitted_tasks_of_worker(int workerid)
{
	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);

	_starpu_barrier_counter_increment(&worker->tasks_barrier);

	return;
}

int starpu_wait_for_all_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
	  return -EDEADLK;
	
	return _starpu_barrier_counter_wait_for_empty_counter(&sched_ctx->tasks_barrier);
}

void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_starpu_barrier_counter_decrement_until_empty_counter(&sched_ctx->tasks_barrier);
}

void _starpu_increment_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	_starpu_barrier_counter_increment(&sched_ctx->tasks_barrier);
}


pthread_mutex_t *_starpu_get_sched_mutex(struct starpu_sched_ctx *sched_ctx, int workerid)
{
	return sched_ctx->sched_mutex[workerid];
}

pthread_cond_t *_starpu_get_sched_cond(struct starpu_sched_ctx *sched_ctx, int workerid)
{
	return sched_ctx->sched_cond[workerid];
}

void starpu_set_sched_ctx(unsigned *sched_ctx)
{
	pthread_setspecific(sched_ctx_key, (void*)sched_ctx);
}

unsigned starpu_get_sched_ctx()
{
	unsigned sched_ctx = *(unsigned*)pthread_getspecific(sched_ctx_key);
	STARPU_ASSERT(sched_ctx >= 0 && sched_ctx < STARPU_NMAX_SCHED_CTXS);
	return sched_ctx;
}

unsigned _starpu_get_nsched_ctxs()
{
	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	return config->topology.nsched_ctxs;
}

void starpu_set_sched_ctx_policy_data(unsigned sched_ctx_id, void* policy_data)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->policy_data = policy_data;
}

void* starpu_get_sched_ctx_policy_data(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->policy_data;
}

void starpu_worker_set_sched_condition(unsigned sched_ctx_id, int workerid, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->sched_mutex[workerid] = sched_mutex;
	sched_ctx->sched_cond[workerid] = sched_cond;
}

void starpu_worker_get_sched_condition(unsigned sched_ctx_id, int workerid, pthread_mutex_t **sched_mutex, pthread_cond_t **sched_cond)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	*sched_mutex = sched_ctx->sched_mutex[workerid];
	*sched_cond = sched_ctx->sched_cond[workerid];
}

void starpu_worker_init_sched_condition(unsigned sched_ctx_id, int workerid)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->sched_mutex[workerid] = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	sched_ctx->sched_cond[workerid] = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
	PTHREAD_MUTEX_INIT(sched_ctx->sched_mutex[workerid], NULL);
	PTHREAD_COND_INIT(sched_ctx->sched_cond[workerid], NULL);
}

void starpu_worker_deinit_sched_condition(unsigned sched_ctx_id, int workerid)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	PTHREAD_MUTEX_DESTROY(sched_ctx->sched_mutex[workerid]);
	PTHREAD_COND_DESTROY(sched_ctx->sched_cond[workerid]);
	free(sched_ctx->sched_mutex[workerid]);
	free(sched_ctx->sched_cond[workerid]);
}

void starpu_create_worker_collection_for_sched_ctx(unsigned sched_ctx_id, int worker_collection_type)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
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


	return;
}

struct worker_collection* starpu_get_worker_collection_of_sched_ctx(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->workers;
}

pthread_mutex_t* starpu_get_changing_ctx_mutex(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return &sched_ctx->changing_ctx_mutex;
}

unsigned starpu_get_nworkers_of_sched_ctx(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	return sched_ctx->workers->nworkers;

}
