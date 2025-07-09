/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016-2016  Uppsala University
 * Copyright (C) 2013-2013  Thibaut Lambert
 * Copyright (C) 2013-2013  Simon Archipoff
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

#include <starpu.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/sched_policy.h>
#include <profiling/starpu_tracing.h>
#include <profiling/profiling.h>
#include <datawizard/memory_nodes.h>
#include <common/barrier.h>
#include <core/debug.h>
#include <core/task.h>
#include <sched_policies/sched_visu.h>

#ifdef HAVE_DLOPEN
#include <dlfcn.h>
#endif

static int use_prefetch = 0;
static double idle[STARPU_NMAXWORKERS];
static double idle_start[STARPU_NMAXWORKERS];

long _starpu_task_break_on_push = -1;
long _starpu_task_break_on_sched = -1;
long _starpu_task_break_on_pop = -1;
long _starpu_task_break_on_exec = -1;
static const char *starpu_idle_file;
static void *dl_sched_handle = NULL;
static const char *sched_lib = NULL;

void _starpu_sched_init(void)
{
	_starpu_visu_init();
	_starpu_task_break_on_push = starpu_getenv_number_default("STARPU_TASK_BREAK_ON_PUSH", -1);
	_starpu_task_break_on_sched = starpu_getenv_number_default("STARPU_TASK_BREAK_ON_SCHED", -1);
	_starpu_task_break_on_pop = starpu_getenv_number_default("STARPU_TASK_BREAK_ON_POP", -1);
	_starpu_task_break_on_exec = starpu_getenv_number_default("STARPU_TASK_BREAK_ON_EXEC", -1);
	starpu_idle_file = starpu_getenv("STARPU_IDLE_FILE");
}

int starpu_get_prefetch_flag(void)
{
	return use_prefetch;
}

static struct starpu_sched_policy *predefined_policies[] =
{
	&_starpu_sched_modular_eager_policy,
	&_starpu_sched_modular_eager_prefetching_policy,
	&_starpu_sched_modular_eager_prio_policy,
	&_starpu_sched_modular_gemm_policy,
	&_starpu_sched_modular_prio_policy,
	&_starpu_sched_modular_prio_prefetching_policy,
	&_starpu_sched_modular_random_policy,
	&_starpu_sched_modular_random_prio_policy,
	&_starpu_sched_modular_random_prefetching_policy,
	&_starpu_sched_modular_random_prio_prefetching_policy,
	&_starpu_sched_modular_parallel_random_policy,
	&_starpu_sched_modular_parallel_random_prio_policy,
	&_starpu_sched_modular_ws_policy,
	&_starpu_sched_modular_dmda_policy,
	&_starpu_sched_modular_dmdap_policy,
	&_starpu_sched_modular_dmdar_policy,
	&_starpu_sched_modular_dmdas_policy,
	&_starpu_sched_modular_heft_policy,
	&_starpu_sched_modular_heft_prio_policy,
	&_starpu_sched_modular_heft2_policy,
	&_starpu_sched_modular_heteroprio_policy,
	&_starpu_sched_modular_heteroprio_heft_policy,
	&_starpu_sched_modular_parallel_heft_policy,
	&_starpu_sched_eager_policy,
	&_starpu_sched_prio_policy,
	&_starpu_sched_random_policy,
	&_starpu_sched_lws_policy,
	&_starpu_sched_ws_policy,
	&_starpu_sched_dm_policy,
	&_starpu_sched_dmda_policy,
	&_starpu_sched_dmda_prio_policy,
	&_starpu_sched_dmda_ready_policy,
	&_starpu_sched_dmda_sorted_policy,
	&_starpu_sched_dmda_sorted_decision_policy,
	&_starpu_sched_parallel_heft_policy,
	&_starpu_sched_peager_policy,
	&_starpu_sched_heteroprio_policy,
	&_starpu_sched_graph_test_policy,
#ifdef STARPU_HAVE_HWLOC
	//&_starpu_sched_tree_heft_hierarchical_policy,
#endif
	NULL
};

static struct starpu_sched_policy *predefined_policies_non_default[] =
{
	&_starpu_sched_darts_policy,
	&_starpu_sched_random_order_policy,
	&_starpu_sched_HFP_policy,
	&_starpu_sched_modular_heft_HFP_policy,
	&_starpu_sched_mst_policy,
	&_starpu_sched_cuthillmckee_policy,
	NULL
};

struct starpu_sched_policy **starpu_sched_get_predefined_policies()
{
	return predefined_policies;
}

struct starpu_sched_policy *_starpu_get_sched_policy(struct _starpu_sched_ctx *sched_ctx)
{
	return sched_ctx->sched_policy;
}

struct starpu_sched_policy *starpu_sched_get_sched_policy_in_ctx(unsigned sched_ctx_id)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_sched_ctx *sched_ctx = &config->sched_ctxs[sched_ctx_id];
	return sched_ctx->sched_policy;
}

struct starpu_sched_policy *starpu_sched_get_sched_policy(void)
{
	unsigned nsched_ctxs = _starpu_get_nsched_ctxs();
	unsigned sched_ctx_id = nsched_ctxs == 1 ? 0 : starpu_sched_ctx_get_context();
	return starpu_sched_get_sched_policy_in_ctx(sched_ctx_id);
}

/*
 *	Methods to initialize the scheduling policy
 */

static void load_sched_policy(struct starpu_sched_policy *sched_policy, struct _starpu_sched_ctx *sched_ctx)
{
	STARPU_ASSERT(sched_policy);

#ifdef STARPU_VERBOSE
	if (sched_policy->policy_name)
	{
		if (sched_policy->policy_description)
			_STARPU_DEBUG("Use %s scheduler (%s)\n", sched_policy->policy_name, sched_policy->policy_description);
		else
			_STARPU_DEBUG("Use %s scheduler \n", sched_policy->policy_name);

	}
#endif

	*(sched_ctx->sched_policy) = *sched_policy;
}

static void load_sched_lib()
{
	/* check if the requested policy can be loaded dynamically */
	sched_lib = starpu_getenv("STARPU_SCHED_LIB");
	if (sched_lib)
	{
#ifdef HAVE_DLOPEN
		if (dl_sched_handle)
		{
			dlclose(dl_sched_handle);
			dl_sched_handle = NULL;
		}
		dl_sched_handle = dlopen(sched_lib, RTLD_NOW);
		if (!dl_sched_handle)
			_STARPU_MSG("Warning: scheduling dynamic library '%s' can not be loaded\n", sched_lib);
#else
		_STARPU_MSG("Environment variable 'STARPU_SCHED_LIB' defined but the dlopen functionality is unavailable on the system\n");
#endif
	}
}

static struct starpu_sched_policy *find_sched_policy_from_name(const char *policy_name)
{
	if (!policy_name)
		return NULL;

	if (strcmp(policy_name, "") == 0)
		return NULL;

	/* check if the requested policy can be loaded dynamically */
	load_sched_lib();
#ifdef HAVE_DLOPEN
	if (dl_sched_handle)
	{
		struct starpu_sched_policy *(*func_sched)(const char *);
		*(void**)(&func_sched) = dlsym(dl_sched_handle, "starpu_get_sched_lib_policy");
		if (!func_sched)
		{
			/* no such symbol */
			_STARPU_MSG("Warning: the library '%s' does not define the function 'starpu_get_sched_lib_policy' (error '%s')\n", sched_lib, dlerror());
			dlclose(dl_sched_handle);
			dl_sched_handle = NULL;
		}
		else
		{
			struct starpu_sched_policy *dl_sched_policy = func_sched(policy_name);
			if (dl_sched_policy)
				return dl_sched_policy;
			else
			{
				dlclose(dl_sched_handle);
				dl_sched_handle = NULL;
			}
		}
	}
#endif

	if (strncmp(policy_name, "heft", 4) == 0)
	{
		_STARPU_MSG("Warning: heft is now called \"dmda\".\n");
		return &_starpu_sched_dmda_policy;
	}

	struct starpu_sched_policy **policy;
	for(policy=predefined_policies ; *policy!=NULL ; policy++)
	{
		struct starpu_sched_policy *p = *policy;
		if (p->policy_name)
		{
			if (strcmp(policy_name, p->policy_name) == 0)
			{
				/* we found a policy with the requested name */
				return p;
			}
		}
	}

	for(policy=predefined_policies_non_default ; *policy!=NULL ; policy++)
	{
		struct starpu_sched_policy *p = *policy;
		if (p->policy_name)
		{
			if (strcmp(policy_name, p->policy_name) == 0)
			{
				/* we found a policy with the requested name */
				return p;
			}
		}
	}

	if (strcmp(policy_name, "help") == 0)
		return NULL;

	_STARPU_MSG("Warning: scheduling policy '%s' was not found, try 'help' to get a list\n", policy_name);

	/* nothing was found */
	return NULL;
}

static void display_sched_help_message(FILE *stream)
{
	const char *sched_env = starpu_getenv("STARPU_SCHED");
	if (sched_env && (strcmp(sched_env, "help") == 0))
	{
		/* display the description of all predefined policies */
		struct starpu_sched_policy **policy;

		fprintf(stream, "\nThe variable STARPU_SCHED can be set to one of the following strings:\n");
		for(policy=predefined_policies ; *policy!=NULL ; policy++)
		{
			struct starpu_sched_policy *p = *policy;
			fprintf(stream, "%-30s\t-> %s\n", p->policy_name, p->policy_description);
		}
		fprintf(stream, "\n");

		for(policy=predefined_policies_non_default ; *policy!=NULL ; policy++)
		{
			struct starpu_sched_policy *p = *policy;
			fprintf(stream, "%-30s\t-> %s\n", p->policy_name, p->policy_description);
		}
		fprintf(stream, "\n");

		load_sched_lib();
#ifdef HAVE_DLOPEN
		if (dl_sched_handle)
		{
			struct starpu_sched_policy **(*func_scheds)(void);
			*(void**)(&func_scheds) = dlsym(dl_sched_handle, "starpu_get_sched_lib_policies");
			if (func_scheds)
			{
				fprintf(stream, "(dynamically available policies)\n");
				struct starpu_sched_policy **dl_sched_policies = func_scheds();
				for(policy=dl_sched_policies ; *policy!=NULL ; policy++)
				{
					struct starpu_sched_policy *p = *policy;
					fprintf(stream, "%-30s\t-> %s\n", p->policy_name, p->policy_description);
				}
				fprintf(stream, "\n");
			}
		}
#endif
	 }
}

struct starpu_sched_policy *_starpu_select_sched_policy(struct _starpu_machine_config *config, const char *required_policy)
{
	struct starpu_sched_policy *selected_policy = NULL;
	struct starpu_conf *user_conf = &config->conf;

	if(required_policy)
		selected_policy = find_sched_policy_from_name(required_policy);

	/* If there is a policy that matches the required name, return it */
	if (selected_policy)
		return selected_policy;

	/* First, we check whether the application explicitly gave a scheduling policy or not */
	if (user_conf && (user_conf->sched_policy))
		return user_conf->sched_policy;

	/* Otherwise, we look if the application specified the name of a policy to load */
	const char *sched_pol_name;
	sched_pol_name = starpu_getenv("STARPU_SCHED");
	if (sched_pol_name == NULL && user_conf && user_conf->sched_policy_name)
		sched_pol_name = user_conf->sched_policy_name;
	if (sched_pol_name)
		selected_policy = find_sched_policy_from_name(sched_pol_name);

	/* If there is a policy that matches the name, return it */
	if (selected_policy)
		return selected_policy;

	/* If no policy was specified, we use the lws policy by default */
	return &_starpu_sched_lws_policy;
}

void _starpu_init_sched_policy(struct _starpu_machine_config *config, struct _starpu_sched_ctx *sched_ctx, struct starpu_sched_policy *selected_policy)
{
	/* Perhaps we have to display some help */
	display_sched_help_message(stderr);

	/* Prefetch is activated by default */
	use_prefetch = starpu_getenv_number("STARPU_PREFETCH");
	if (use_prefetch == -1)
		use_prefetch = 1;

	/* Set calibrate flag */
	_starpu_set_calibrate_flag(config->conf.calibrate);

	load_sched_policy(selected_policy, sched_ctx);

	if (starpu_getenv_number_default("STARPU_WORKER_TREE", 0))
	{
#ifdef STARPU_HAVE_HWLOC
		sched_ctx->sched_policy->worker_type = STARPU_WORKER_TREE;
#else
		_STARPU_DISP("STARPU_WORKER_TREE ignored, please rebuild StarPU with hwloc support to enable it.\n");
#endif
	}
	starpu_sched_ctx_create_worker_collection(sched_ctx->id,
						  sched_ctx->sched_policy->worker_type);

	_STARPU_SCHED_BEGIN;
	sched_ctx->sched_policy->init_sched(sched_ctx->id);
	_STARPU_SCHED_END;
}

void _starpu_deinit_sched_policy(struct _starpu_sched_ctx *sched_ctx)
{
	struct starpu_sched_policy *policy = sched_ctx->sched_policy;
	if (policy->deinit_sched)
	{
		_STARPU_SCHED_BEGIN;
		policy->deinit_sched(sched_ctx->id);
		_STARPU_SCHED_END;
	}
	starpu_sched_ctx_delete_worker_collection(sched_ctx->id);
#ifdef HAVE_DLOPEN
	if (dl_sched_handle)
	{
		dlclose(dl_sched_handle);
		dl_sched_handle = NULL;
	}
#endif
}

void _starpu_sched_task_submit(struct starpu_task *task)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(task->sched_ctx);
	if (!sched_ctx->sched_policy)
		return;
	if (!sched_ctx->sched_policy->submit_hook)
		return;
	_STARPU_SCHED_BEGIN;
	sched_ctx->sched_policy->submit_hook(task);
	_STARPU_SCHED_END;
}

void _starpu_sched_do_schedule(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if (!sched_ctx->sched_policy)
		return;
	if (!sched_ctx->sched_policy->do_schedule)
		return;
	_STARPU_SCHED_BEGIN;
	sched_ctx->sched_policy->do_schedule(sched_ctx_id);
	_STARPU_SCHED_END;
}

void _starpu_sched_reset_scheduler(unsigned sched_ctx_id)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if (sched_ctx->sched_policy && sched_ctx->sched_policy->reset_scheduler)
		sched_ctx->sched_policy->reset_scheduler(sched_ctx_id);
}

static void _starpu_push_task_on_specific_worker_notify_sched(struct starpu_task *task, struct _starpu_worker *worker, int workerid, int perf_workerid)
{
	/* if we push a task on a specific worker, notify all the sched_ctxs the worker belongs to */
	struct _starpu_sched_ctx_list_iterator list_it;

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		struct _starpu_sched_ctx_elt *e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(e->sched_ctx);
		if (sched_ctx->sched_policy != NULL && sched_ctx->sched_policy->push_task_notify)
		{
			_STARPU_SCHED_BEGIN;
			sched_ctx->sched_policy->push_task_notify(task, workerid, perf_workerid, sched_ctx->id);
			_STARPU_SCHED_END;
		}
	}
}

/* Enqueue a task into the list of tasks explicitly attached to a worker. In
 * case workerid identifies a combined worker, a task will be enqueued into
 * each worker of the combination. */
static int _starpu_push_task_on_specific_worker(struct starpu_task *task, int workerid)
{
	int nbasic_workers = (int)starpu_worker_get_count();

	// We push on a specific worker, and so we will not pass by the unpartition engine, so we need to turn it NOW
#ifdef STARPU_RECURSIVE_TASKS
	if (_starpu_turn_task_into_recursive_task_at_scheduler(task))
	{
		return 0;
	}
#endif

	/* Is this a basic worker or a combined worker ? */
	int is_basic_worker = (workerid < nbasic_workers);

	struct _starpu_worker *worker = NULL;
	struct _starpu_combined_worker *combined_worker = NULL;

	if (is_basic_worker)
	{
		worker = _starpu_get_worker_struct(workerid);
	}
	else
	{
		combined_worker = _starpu_get_combined_worker_struct(workerid);
	}

	if (use_prefetch && task->prefetched == 0)
		starpu_prefetch_task_input_for(task, workerid);

	if (is_basic_worker)
		_starpu_push_task_on_specific_worker_notify_sched(task, worker, workerid, workerid);
	else
	{
		/* Notify all workers of the combined worker */
		int worker_size = combined_worker->worker_size;
		int *combined_workerid = combined_worker->combined_workerid;

		int j;
		for (j = 0; j < worker_size; j++)
		{
			int subworkerid = combined_workerid[j];
			_starpu_push_task_on_specific_worker_notify_sched(task, _starpu_get_worker_struct(subworkerid), subworkerid, workerid);
		}
	}

#ifdef STARPU_USE_SC_HYPERVISOR
	starpu_sched_ctx_call_pushed_task_cb(workerid, task->sched_ctx);
#endif //STARPU_USE_SC_HYPERVISOR
	if (is_basic_worker)
	{
		unsigned node = starpu_worker_get_memory_node(workerid);
		if (_starpu_task_uses_multiformat_handles(task))
		{
			unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
			unsigned i;
			for (i = 0; i < nbuffers; i++)
			{
				struct starpu_task *conversion_task;
				starpu_data_handle_t handle;

				handle = STARPU_TASK_GET_HANDLE(task, i);
				if (!_starpu_handle_needs_conversion_task(handle, node))
					continue;

				conversion_task = _starpu_create_conversion_task(handle, node);
				conversion_task->mf_skip = 1;
				conversion_task->execute_on_a_specific_worker = 1;
				conversion_task->workerid = workerid;
				_starpu_task_submit_conversion_task(conversion_task, workerid);
				//_STARPU_DEBUG("Pushing a conversion task\n");
			}

			for (i = 0; i < nbuffers; i++)
			{
				starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
				handle->mf_node = node;
			}
		}
//		if(task->sched_ctx != _starpu_get_initial_sched_ctx()->id)

		return _starpu_push_local_task(worker, task);
	}
	else
	{
		/* This is a combined worker so we create task aliases */
		int worker_size = combined_worker->worker_size;
		int *combined_workerid = combined_worker->combined_workerid;

		int ret = 0;

		struct _starpu_job *job = _starpu_get_job_associated_to_task(task);
		job->task_size = worker_size;
		job->combined_workerid = workerid;
		job->active_task_alias_count = 0;

		STARPU_PTHREAD_BARRIER_INIT(&job->before_work_barrier, NULL, worker_size);
		STARPU_PTHREAD_BARRIER_INIT(&job->after_work_barrier, NULL, worker_size);
		job->after_work_busy_barrier = worker_size;

		/* Note: we have to call that early, or else the task may have
		 * disappeared already */
		starpu_push_task_end(task);

		int j;
		for (j = 0; j < worker_size; j++)
		{
			struct starpu_task *alias = starpu_task_dup(task);
			alias->destroy = 1;

			_starpu_trace_job_push(alias, alias->priority);
			worker = _starpu_get_worker_struct(combined_workerid[j]);
			ret |= _starpu_push_local_task(worker, alias);
		}

		return ret;
	}
}

/* the generic interface that call the proper underlying implementation */

int _starpu_push_task(struct _starpu_job *j)
{
#ifdef STARPU_SIMGRID
	if (_starpu_simgrid_task_push_cost())
		starpu_sleep(0.000001);
#endif
	if(j->task->prologue_callback_func)
	{
		_starpu_set_current_task(j->task);
		j->task->prologue_callback_func(j->task->prologue_callback_arg);
		_starpu_set_current_task(NULL);
	}

	if (j->task->transaction)
	{
		/* If task is part of a transaction and its epoch is cancelled, switch its
		 * 'where' field to STARPU_NOWHERE to skip its execution */
		struct starpu_transaction *p_trs = j->task->transaction;
		STARPU_ASSERT(j->task->transaction->state == _starpu_trs_initialized);
		_starpu_spin_lock(&p_trs->lock);
		STARPU_ASSERT(!_starpu_trs_epoch_list_empty(&p_trs->epoch_list));
		struct _starpu_trs_epoch *p_epoch = _starpu_trs_epoch_list_front(&p_trs->epoch_list);
		STARPU_ASSERT(p_epoch == j->task->trs_epoch);
		STARPU_ASSERT(p_epoch->state == _starpu_trs_epoch_confirmed || p_epoch->state == _starpu_trs_epoch_cancelled);
		if (p_epoch->state == _starpu_trs_epoch_cancelled)
		{
			j->task->where = STARPU_NOWHERE;
		}
		_starpu_spin_unlock(&p_trs->lock);
	}

	return _starpu_repush_task(j);
}

int _starpu_repush_task(struct _starpu_job *j)
{
	struct starpu_task *task = j->task;
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(task->sched_ctx);
	int ret;

	_STARPU_LOG_IN();

	unsigned can_push = _starpu_increment_nready_tasks_of_sched_ctx(task->sched_ctx, task->flops, task);
	STARPU_ASSERT(task->status == STARPU_TASK_BLOCKED || task->status == STARPU_TASK_BLOCKED_ON_TAG || task->status == STARPU_TASK_BLOCKED_ON_TASK || task->status == STARPU_TASK_BLOCKED_ON_DATA);
	task->status = STARPU_TASK_READY;
	_STARPU_RECURSIVE_TASKS_DEBUG("[%d] %s(%p) is now READY.\n", starpu_worker_get_id(), starpu_task_get_name(task), task);
	const unsigned continuation =
#ifdef STARPU_OPENMP
		j->continuation
#else
		0
#endif
		;
	if (!_starpu_perf_counter_paused() && !j->internal && !continuation)
	{
		(void) STARPU_PERF_COUNTER_ADD64(& _starpu_task__g_current_submitted__value, -1);
		int64_t value = STARPU_PERF_COUNTER_ADD64(& _starpu_task__g_current_ready__value, 1);
		_starpu_perf_counter_update_max_int64(&_starpu_task__g_peak_ready__value, value);
		if (task->cl && task->cl->perf_counter_values)
		{
			struct starpu_perf_counter_sample_cl_values * const pcv = task->cl->perf_counter_values;

			(void)STARPU_PERF_COUNTER_ADD64(&pcv->task.current_submitted, -1);
			value = STARPU_PERF_COUNTER_ADD64(&pcv->task.current_ready, 1);
			_starpu_perf_counter_update_max_int64(&pcv->task.peak_ready, value);
		}
	}
	STARPU_AYU_ADDTOTASKQUEUE(j->job_id, -1);
	/* if the context does not have any workers save the tasks in a temp list */
	if ((task->cl != NULL && task->where != STARPU_NOWHERE) && (!sched_ctx->is_initial_sched))
	{
		/*if there are workers in the ctx that are not able to execute tasks
		  we consider the ctx empty */
		unsigned able = _starpu_workers_able_to_execute_task(task, sched_ctx);

		if(!able)
		{
			_starpu_sched_ctx_lock_write(sched_ctx->id);
			starpu_task_list_push_front(&sched_ctx->empty_ctx_tasks, task);
			_starpu_sched_ctx_unlock_write(sched_ctx->id);
#ifdef STARPU_USE_SC_HYPERVISOR
			if(sched_ctx->id != 0 && sched_ctx->perf_counters != NULL
			   && sched_ctx->perf_counters->notify_empty_ctx)
			{
				_starpu_trace_hypervisor_begin();
				sched_ctx->perf_counters->notify_empty_ctx(sched_ctx->id, task);
				_starpu_trace_hypervisor_end();
			}
#endif
			return 0;
		}

	}

	if(!can_push)
		return 0;
	/* in case there is no codelet associated to the task (that's a control
	 * task), we directly execute its callback and enforce the
	 * corresponding dependencies */
	if (task->cl == NULL || task->where == STARPU_NOWHERE)
	{
		_starpu_trace_task_name_line_color(j);
		if (!_starpu_perf_counter_paused() && !j->internal)
		{
			(void)STARPU_PERF_COUNTER_ADD64(& _starpu_task__g_current_ready__value, -1);
			if (task->cl && task->cl->perf_counter_values)
			{
				struct starpu_perf_counter_sample_cl_values * const pcv = task->cl->perf_counter_values;
				(void)STARPU_PERF_COUNTER_ADD64(&pcv->task.current_ready, -1);
			}
		}
		task->status = STARPU_TASK_RUNNING;
		if (task->prologue_callback_pop_func)
		{
			_starpu_set_current_task(task);
			task->prologue_callback_pop_func(task->prologue_callback_pop_arg);
			_starpu_set_current_task(NULL);
		}
#ifdef STARPU_RECURSIVE_TASKS
		// We need to unpartition the task maybe
		if (_starpu_turn_task_into_recursive_task_at_scheduler(task))
		{
			return 0; // we need to wait for unpartition !
		}
#endif

		{
			int worker_id = starpu_worker_get_id();
			_starpu_trace_start_codelet_body(j, 0, NULL, worker_id, 0);
			_starpu_trace_end_codelet_body(j, 0, NULL, worker_id, 0);
		}

		if (task->cl && task->cl->specific_nodes)
		{
			/* Nothing to do, but we are asked to fetch data on some memory nodes */
			_starpu_fetch_nowhere_task_input(j);
		}
		else
		{
			if (task->cl
#ifdef STARPU_RECURSIVE_TASKS
//			    && !j->is_recursive_task
#endif
			    )
				__starpu_push_task_output(j);
			_starpu_handle_job_termination(j);
			_STARPU_LOG_OUT_TAG("handle_job_termination");
		}
		return 0;
	}

	ret = _starpu_push_task_to_workers(task);
	if (ret == -EAGAIN)
		/* pushed to empty context, that's fine */
		ret = 0;
	return ret;
}

int _starpu_push_task_to_workers(struct starpu_task *task)
{
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(task->sched_ctx);

	_starpu_trace_job_push(task, task->priority);

	/* if the contexts still does not have workers put the task back to its place in
	   the empty ctx list */
	if(!sched_ctx->is_initial_sched)
	{
		/*if there are workers in the ctx that are not able to execute tasks
		  we consider the ctx empty */
		unsigned able = _starpu_workers_able_to_execute_task(task, sched_ctx);

		if (!able)
		{
			_starpu_sched_ctx_lock_write(sched_ctx->id);
			starpu_task_list_push_back(&sched_ctx->empty_ctx_tasks, task);
			_starpu_sched_ctx_unlock_write(sched_ctx->id);
#ifdef STARPU_USE_SC_HYPERVISOR
			if(sched_ctx->id != 0 && sched_ctx->perf_counters != NULL
			   && sched_ctx->perf_counters->notify_empty_ctx)
			{
				_starpu_trace_hypervisor_begin();
				sched_ctx->perf_counters->notify_empty_ctx(sched_ctx->id, task);
				_starpu_trace_hypervisor_end();
			}
#endif

			return -EAGAIN;
		}
	}

	_starpu_profiling_set_task_push_start_time(task);

	int ret = 0;
	if (STARPU_UNLIKELY(task->execute_on_a_specific_worker))
	{
		ret = _starpu_push_task_on_specific_worker(task, task->workerid);
	}
	else
	{
		struct _starpu_machine_config *config = _starpu_get_machine_config();

		if(!sched_ctx->sched_policy)
		{
			/* Note: we have to call that early, or else the task may have
			 * disappeared already */
			starpu_push_task_end(task);
			if(!sched_ctx->awake_workers)
				ret = _starpu_push_task_on_specific_worker(task, sched_ctx->main_primary);
			else
			{
				struct starpu_worker_collection *workers = sched_ctx->workers;

				struct _starpu_job *job = _starpu_get_job_associated_to_task(task);
				job->task_size = workers->nworkers;
				job->combined_workerid = -1; // workerid; its a ctx not combined worker
				job->active_task_alias_count = 0;

				STARPU_PTHREAD_BARRIER_INIT(&job->before_work_barrier, NULL, workers->nworkers);
				STARPU_PTHREAD_BARRIER_INIT(&job->after_work_barrier, NULL, workers->nworkers);
				job->after_work_busy_barrier = workers->nworkers;

				struct starpu_sched_ctx_iterator it;
				if(workers->init_iterator)
					workers->init_iterator(workers, &it);

				while(workers->has_next(workers, &it))
				{
					unsigned workerid = workers->get_next(workers, &it);
					struct starpu_task *alias;
					if (job->task_size > 1)
					{
						alias = starpu_task_dup(task);
						_starpu_trace_job_push(alias, alias->priority);
						alias->destroy = 1;
					}
					else
						alias = task;
					ret |= _starpu_push_task_on_specific_worker(alias, workerid);
				}
			}
		}
		else
		{
			/* When a task can only be executed on a given arch and we have
			 * only one memory node for that arch, we can systematically
			 * prefetch before the scheduling decision. */
			if (!sched_ctx->sched_policy->prefetches
				&& starpu_get_prefetch_flag()
				&& starpu_memory_nodes_get_count() > 1)
			{
				enum starpu_worker_archtype type;
				for (type = 0; type < STARPU_NARCH; type++)
				{
					if (task->where == (int32_t) STARPU_WORKER_TO_MASK(type))
					{
						if (config->arch_nodeid[type] >= 0)
							starpu_prefetch_task_input_on_node(task, config->arch_nodeid[type]);
						break;
					}
				}
			}

			STARPU_ASSERT(sched_ctx->sched_policy->push_task);
			/* check out if there are any workers in the context */
			unsigned nworkers = starpu_sched_ctx_get_nworkers(sched_ctx->id);
			if (nworkers == 0)
				ret = -1;
			else
			{
				struct _starpu_worker *worker = _starpu_get_local_worker_key();
				if (worker)
				{
					STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
					_starpu_worker_enter_sched_op(worker);
					STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
				}
				_STARPU_TASK_BREAK_ON(task, push);
				_STARPU_SCHED_BEGIN;
				ret = sched_ctx->sched_policy->push_task(task);
				_STARPU_SCHED_END;
				if (worker)
				{
					STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
					_starpu_worker_leave_sched_op(worker);
					STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
				}
			}
		}

		if(ret == -1)
		{
			_STARPU_MSG("repush task \n");
			_starpu_trace_job_pop(task, task->priority);
			ret = _starpu_push_task_to_workers(task);
		}
	}
	/* Note: from here, the task might have been destroyed already! */
	_STARPU_LOG_OUT();
	return ret;

}

/* This is called right after the scheduler has pushed a task to a queue
 * but just before releasing mutexes: we need the task to still be alive!
 */
int starpu_push_task_end(struct starpu_task *task)
{
	if (task->where == STARPU_CPU && !_starpu_task_is_recursive(task))
	{
//		fprintf(stderr, "CPu has task %p %s at %lf\n", task, task->name, starpu_timing_now());
	}

	_starpu_profiling_set_task_push_end_time(task);
	task->scheduled = 1;
	return 0;
}

/* This is called right after the scheduler has pushed a task to a queue
 * but just before releasing mutexes: we need the task to still be alive!
 */
int _starpu_pop_task_end(struct starpu_task *task)
{
	if (!task)
		return 0;
	_starpu_trace_job_pop(task, task->priority);
	return 0;
}

/*
 * Given a handle that needs to be converted in order to be used on the given
 * node, returns a task that takes care of the conversion.
 */
struct starpu_task *_starpu_create_conversion_task(starpu_data_handle_t handle, unsigned int node)
{
	return _starpu_create_conversion_task_for_arch(handle, starpu_node_get_kind(node));
}

struct starpu_task *_starpu_create_conversion_task_for_arch(starpu_data_handle_t handle, enum starpu_node_kind node_kind)
{
	struct starpu_task *conversion_task;

/* Driver porters: adding your driver here is optional, only needed for the support of multiple formats.  */

#if defined(STARPU_USE_OPENCL) || defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	struct starpu_multiformat_interface *format_interface;
#endif

	conversion_task = starpu_task_create();
	conversion_task->name = "conversion_task";
	conversion_task->synchronous = 0;
	STARPU_TASK_SET_HANDLE(conversion_task, handle, 0);

#if defined(STARPU_USE_OPENCL) || defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	/* The node does not really matter here */
	format_interface = (struct starpu_multiformat_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
#endif

	_starpu_spin_lock(&handle->header_lock);
	handle->refcnt++;
	_STARPU_RECURSIVE_TASKS_DEBUG("Take refcnt on data %p by conversion task\n", handle);
	_STARPU_RECURSIVE_TASKS_DEBUG("Take busy count on data %p by conversion task\n", handle);
	handle->busy_count++;
	_starpu_spin_unlock(&handle->header_lock);

	/* Driver porters: adding your driver here is optional, only needed for the support of multiple formats.  */

	switch(node_kind)
	{
	case STARPU_CPU_RAM:
		switch (starpu_node_get_kind(handle->mf_node))
		{
		case STARPU_CPU_RAM:
			STARPU_ABORT();
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
		case STARPU_CUDA_RAM:
		{
			struct starpu_multiformat_data_interface_ops *mf_ops;
			mf_ops = (struct starpu_multiformat_data_interface_ops *) handle->ops->get_mf_ops(format_interface);
			conversion_task->cl = mf_ops->cuda_to_cpu_cl;
			break;
		}
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
		case STARPU_OPENCL_RAM:
		{
			struct starpu_multiformat_data_interface_ops *mf_ops;
			mf_ops = (struct starpu_multiformat_data_interface_ops *) handle->ops->get_mf_ops(format_interface);
			conversion_task->cl = mf_ops->opencl_to_cpu_cl;
			break;
		}
#endif
		default:
			_STARPU_ERROR("Oops : %u\n", handle->mf_node);
		}
		break;
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	case STARPU_CUDA_RAM:
		{
			struct starpu_multiformat_data_interface_ops *mf_ops;
			mf_ops = (struct starpu_multiformat_data_interface_ops *) handle->ops->get_mf_ops(format_interface);
			conversion_task->cl = mf_ops->cpu_to_cuda_cl;
			break;
		}
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	case STARPU_OPENCL_RAM:
	{
		struct starpu_multiformat_data_interface_ops *mf_ops;
		mf_ops = (struct starpu_multiformat_data_interface_ops *) handle->ops->get_mf_ops(format_interface);
		conversion_task->cl = mf_ops->cpu_to_opencl_cl;
		break;
	}
#endif
	default:
		STARPU_ABORT();
	}

	_starpu_codelet_check_deprecated_fields(conversion_task->cl);
	STARPU_TASK_SET_MODE(conversion_task, STARPU_RW, 0);
	return conversion_task;
}

static
struct _starpu_sched_ctx* _get_next_sched_ctx_to_pop_into(struct _starpu_worker *worker)
{
	struct _starpu_sched_ctx_elt *e = NULL;
	struct _starpu_sched_ctx_list_iterator list_it;
	int found = 0;

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		if (e->task_number > 0)
			return _starpu_get_sched_ctx_struct(e->sched_ctx);
	}

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		if (e->last_poped)
		{
			e->last_poped = 0;
			if (_starpu_sched_ctx_list_iterator_has_next(&list_it))
			{
				e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
				found = 1;
			}
			break;
		}
	}
	if (!found)
		e = worker->sched_ctx_list->head;
	e->last_poped = 1;

	return _starpu_get_sched_ctx_struct(e->sched_ctx);
}

struct starpu_task *_starpu_pop_task(struct _starpu_worker *worker)
{
	struct starpu_task *task;
	int worker_id;
	unsigned node;

	/* We can't tell in advance which task will be picked up, so we measure
	 * a timestamp, and will attribute it afterwards to the task. */
	int profiling = starpu_profiling_status_get();
	struct timespec pop_start_time;
	if (profiling)
		_starpu_clock_gettime(&pop_start_time);

pick:
	/* perhaps there is some local task to be executed first */
	task = _starpu_pop_local_task(worker);

	if (task)
		_STARPU_TASK_BREAK_ON(task, pop);

	/* get tasks from the stacks of the strategy */
	if(!task)
	{
		struct _starpu_sched_ctx *sched_ctx;
#ifndef STARPU_NON_BLOCKING_DRIVERS
		int been_here[STARPU_NMAX_SCHED_CTXS];
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
			been_here[i] = 0;

		while(!task)
#endif
		{
			if(worker->nsched_ctxs == 1)
				sched_ctx = _starpu_get_initial_sched_ctx();
			else
			{
				while(1)
				{
					/** Caution
					 * If you use multiple contexts your scheduler *needs*
					 * to update the variable task_number of the ctx list.
					 * In order to get the best performances.
					 * This is done using functions :
					 *   starpu_sched_ctx_list_task_counters_increment...(...)
					 *   starpu_sched_ctx_list_task_counters_decrement...(...)
					**/
					sched_ctx = _get_next_sched_ctx_to_pop_into(worker);

					if(worker->removed_from_ctx[sched_ctx->id] == 1 && worker->shares_tasks_lists[sched_ctx->id] == 1)
					{
						_starpu_worker_gets_out_of_ctx(sched_ctx->id, worker);
						worker->removed_from_ctx[sched_ctx->id] = 0;
						sched_ctx = NULL;
					}
					else
						break;
				}
			}

			if(sched_ctx && sched_ctx->id != STARPU_NMAX_SCHED_CTXS)
			{
				if (sched_ctx->sched_policy && sched_ctx->sched_policy->pop_task)
				{
					/* Note: we do not push the scheduling state here, because
					 * otherwise when a worker is idle, we'd keep
					 * pushing/popping a scheduling state here, while what we
					 * want to see in the trace is a permanent idle state. */
					task = sched_ctx->sched_policy->pop_task(sched_ctx->id);
					if (task)
						_STARPU_TASK_BREAK_ON(task, pop);
					_starpu_pop_task_end(task);
				}
			}

			if(!task)
			{
				/* it doesn't matter if it shares tasks list or not in the scheduler,
				   if it does not have any task to pop just get it out of here */
				/* however if it shares a task list it will be removed as soon as he
				  finishes this job (in handle_job_termination) */
				if(worker->removed_from_ctx[sched_ctx->id])
				{
					_starpu_worker_gets_out_of_ctx(sched_ctx->id, worker);
					worker->removed_from_ctx[sched_ctx->id] = 0;
				}
#ifdef STARPU_USE_SC_HYPERVISOR
				if(worker->pop_ctx_priority)
				{
					struct starpu_sched_ctx_performance_counters *perf_counters = sched_ctx->perf_counters;
					if(sched_ctx->id != 0 && perf_counters != NULL && perf_counters->notify_idle_cycle && _starpu_sched_ctx_allow_hypervisor(sched_ctx->id))
					{
//					_starpu_trace_hypervisor_begin();
						perf_counters->notify_idle_cycle(sched_ctx->id, worker->workerid, 1.0);
//					_starpu_trace_hypervisor_end();
					}
				}
#endif //STARPU_USE_SC_HYPERVISOR

#ifndef STARPU_NON_BLOCKING_DRIVERS
				if(been_here[sched_ctx->id] || worker->nsched_ctxs == 1)
					break;

				been_here[sched_ctx->id] = 1;

#endif
			}
		}
	  }


	if (!task)
	{
		if (starpu_idle_file)
			idle_start[worker->workerid] = starpu_timing_now();
		return NULL;
	}

	if(starpu_idle_file && idle_start[worker->workerid] != 0.0)
	{
		double idle_end = starpu_timing_now();
		idle[worker->workerid] += (idle_end - idle_start[worker->workerid]);
		idle_start[worker->workerid] = 0.0;
	}


#ifdef STARPU_USE_SC_HYPERVISOR
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(task->sched_ctx);
	struct starpu_sched_ctx_performance_counters *perf_counters = sched_ctx->perf_counters;

	if(sched_ctx->id != 0 && perf_counters != NULL && perf_counters->notify_poped_task && _starpu_sched_ctx_allow_hypervisor(sched_ctx->id))
	{
//		_starpu_trace_hypervisor_begin();
		perf_counters->notify_poped_task(task->sched_ctx, worker->workerid);
//		_starpu_trace_hypervisor_end();
	}
#endif //STARPU_USE_SC_HYPERVISOR


	/* Make sure we do not bother with all the multiformat-specific code if
	 * it is not necessary. */
	if (!_starpu_task_uses_multiformat_handles(task))
		goto profiling;


	/* This is either a conversion task, or a regular task for which the
	 * conversion tasks have already been created and submitted */
	if (task->mf_skip)
		goto profiling;

	/*
	 * This worker may not be able to execute this task. In this case, we
	 * should return the task anyway. It will be pushed back almost immediately.
	 * This way, we avoid computing and executing the conversions tasks.
	 * Here, we do not care about what implementation is used.
	 */
	worker_id = starpu_worker_get_id_check();
	if (!starpu_worker_can_execute_task_first_impl(worker_id, task, NULL))
		return task;

	node = starpu_worker_get_memory_node(worker_id);

	/*
	 * We do have a task that uses multiformat handles. Let's create the
	 * required conversion tasks.
	 */
	unsigned i;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	for (i = 0; i < nbuffers; i++)
	{
		struct starpu_task *conversion_task;
		starpu_data_handle_t handle;

		handle = STARPU_TASK_GET_HANDLE(task, i);
		if (!_starpu_handle_needs_conversion_task(handle, node))
			continue;
		conversion_task = _starpu_create_conversion_task(handle, node);
		conversion_task->mf_skip = 1;
		conversion_task->execute_on_a_specific_worker = 1;
		conversion_task->workerid = worker_id;
		/*
		 * Next tasks will need to know where these handles have gone.
		 */
		handle->mf_node = node;
		_starpu_task_submit_conversion_task(conversion_task, worker_id);
	}

	task->mf_skip = 1;
	starpu_task_prio_list_push_back(&worker->local_tasks, task);
	goto pick;

profiling:
	if (profiling)
	{
		struct starpu_profiling_task_info *profiling_info;
		profiling_info = task->profiling_info;

		/* The task may have been created before profiling was enabled,
		 * so we check if the profiling_info structure is available
		 * even though we already tested if profiling is enabled. */
		if (profiling_info)
		{
			profiling_info->pop_start_time = pop_start_time;
			_starpu_clock_gettime(&profiling_info->pop_end_time);
		}
	}

	if(task->prologue_callback_pop_func)
	{
		_starpu_set_current_task(task);
		task->prologue_callback_pop_func(task->prologue_callback_pop_arg);
		_starpu_set_current_task(NULL);
	}

	_sched_visu_pop_ready_task(task);

	return task;
}

void _starpu_sched_pre_exec_hook(struct starpu_task *task)
{
	unsigned sched_ctx_id = starpu_sched_ctx_get_ctx_for_task(task);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	_sched_visu_get_current_tasks_for_visualization(task, sched_ctx_id);

	if (sched_ctx->sched_policy && sched_ctx->sched_policy->pre_exec_hook)
	{
		_STARPU_SCHED_BEGIN;
		sched_ctx->sched_policy->pre_exec_hook(task, sched_ctx_id);
		_STARPU_SCHED_END;
	}

	if(!sched_ctx->sched_policy)
	{
		int workerid = starpu_worker_get_id();
		struct _starpu_worker *worker =  _starpu_get_worker_struct(workerid);
		struct _starpu_sched_ctx_list_iterator list_it;

		_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
		while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
		{
			struct _starpu_sched_ctx *other_sched_ctx;
			struct _starpu_sched_ctx_elt *e;

			e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
			other_sched_ctx = _starpu_get_sched_ctx_struct(e->sched_ctx);
			if (other_sched_ctx != sched_ctx &&
			    other_sched_ctx->sched_policy != NULL &&
			    other_sched_ctx->sched_policy->pre_exec_hook)
			{
				_STARPU_SCHED_BEGIN;
				other_sched_ctx->sched_policy->pre_exec_hook(task, other_sched_ctx->id);
				_STARPU_SCHED_END;
			}
		}
	}

}

void _starpu_sched_post_exec_hook(struct starpu_task *task)
{
	STARPU_ASSERT(task->cl != NULL && task->cl->where != STARPU_NOWHERE);
	unsigned sched_ctx_id = starpu_sched_ctx_get_ctx_for_task(task);
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if (sched_ctx->sched_policy && sched_ctx->sched_policy->post_exec_hook)
	{
		_STARPU_SCHED_BEGIN;
		sched_ctx->sched_policy->post_exec_hook(task, sched_ctx_id);
		_STARPU_SCHED_END;
	}
	if(!sched_ctx->sched_policy)
	{
		int workerid = starpu_worker_get_id();
		struct _starpu_worker *worker =  _starpu_get_worker_struct(workerid);
		struct _starpu_sched_ctx_list_iterator list_it;

		_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
		while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
		{
			struct _starpu_sched_ctx *other_sched_ctx;
			struct _starpu_sched_ctx_elt *e;

			e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
			other_sched_ctx = _starpu_get_sched_ctx_struct(e->sched_ctx);
			if (other_sched_ctx != sched_ctx &&
			    other_sched_ctx->sched_policy != NULL &&
			    other_sched_ctx->sched_policy->post_exec_hook)
			{
				_STARPU_SCHED_BEGIN;
				other_sched_ctx->sched_policy->post_exec_hook(task, other_sched_ctx->id);
				_STARPU_SCHED_END;
			}
		}
	}
}

int starpu_push_local_task(int workerid, struct starpu_task *task, int back STARPU_ATTRIBUTE_UNUSED)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);

	return  _starpu_push_local_task(worker, task);
}

void _starpu_print_idle_time()
{
	if(!starpu_idle_file)
		return;
	double all_idle = 0.0;
	int i = 0;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		all_idle += idle[i];

	FILE *f;
	f = fopen(starpu_idle_file, "a");
	if (!f)
	{
		_STARPU_MSG("couldn't open %s: %s\n", starpu_idle_file, strerror(errno));
	}
	else
	{
		fprintf(f, "%lf \n", all_idle);
		fclose(f);
	}
}

void starpu_sched_task_break(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TASK_BREAK_ON(task, sched);
}
