/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2016       Uppsala University
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
#include <starpu_profiling.h>
#include <common/config.h>
#include <common/utils.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>
#include <core/task.h>
#include <float.h>
#include <dirent.h>

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

static int _starpu_expected_transfer_time_writeback;

void _starpu_init_perfmodel(void)
{
	_starpu_expected_transfer_time_writeback = starpu_getenv_number_default("STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK", 0);
}

/* This flag indicates whether performance models should be calibrated or not.
 *	0: models need not be calibrated
 *	1: models must be calibrated
 *	2: models must be calibrated, existing models are overwritten.
 */
static unsigned calibrate_flag = 0;
void _starpu_set_calibrate_flag(unsigned val)
{
	calibrate_flag = val;
}

unsigned _starpu_get_calibrate_flag(void)
{
	return calibrate_flag;
}

struct starpu_perfmodel_arch* starpu_worker_get_perf_archtype(int workerid, unsigned sched_ctx_id)
{
	STARPU_ASSERT(workerid>=0);

	if(sched_ctx_id != STARPU_NMAX_SCHED_CTXS)
	{
		unsigned child_sched_ctx = starpu_sched_ctx_worker_is_master_for_child_ctx(workerid, sched_ctx_id);
		if(child_sched_ctx != STARPU_NMAX_SCHED_CTXS)
			return _starpu_sched_ctx_get_perf_archtype(child_sched_ctx);
		struct _starpu_sched_ctx *stream_ctx = _starpu_worker_get_ctx_stream(workerid);
		if(stream_ctx != NULL)
			return _starpu_sched_ctx_get_perf_archtype(stream_ctx->id);
	}

	struct _starpu_machine_config *config = _starpu_get_machine_config();

	/* This workerid may either be a basic worker or a combined worker */
	unsigned nworkers = config->topology.nworkers;

	if (workerid < (int)config->topology.nworkers)
		return &config->workers[workerid].perf_arch;


	/* We have a combined worker */
	unsigned ncombinedworkers = config->topology.ncombinedworkers;
	STARPU_ASSERT(workerid < (int)(ncombinedworkers + nworkers));
	return &config->combined_workers[workerid - nworkers].perf_arch;
}

/*
 * PER WORKER model
 */

static double per_worker_task_expected_perf(struct starpu_perfmodel *model, unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	double (*worker_cost_function)(struct starpu_task *task, unsigned workerid, unsigned nimpl);

	worker_cost_function = model->worker_cost_function;
	STARPU_ASSERT_MSG(worker_cost_function, "STARPU_PER_WORKER needs worker_cost_function to be defined");

	return worker_cost_function(task, workerid, nimpl);
}

/*
 * PER ARCH model
 */

static double per_arch_task_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch * arch, struct starpu_task *task, unsigned nimpl)
{
	int comb;
	double (*per_arch_cost_function)(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);

	if (model->arch_cost_function)
		return model->arch_cost_function(task, arch, nimpl);

	comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	STARPU_ASSERT_MSG(comb != -1, "Didn't find the proper arch combination\n");
	STARPU_ASSERT_MSG(model->state->per_arch[comb] != NULL, "STARPU_PER_ARCH needs per-arch cost_function to be defined");

	per_arch_cost_function = model->state->per_arch[comb][nimpl].cost_function;
	STARPU_ASSERT_MSG(per_arch_cost_function, "STARPU_PER_ARCH needs per-arch cost_function to be defined");

	return per_arch_cost_function(task, arch, nimpl);
}

/*
 * Common model
 */

double starpu_worker_get_relative_speedup(struct starpu_perfmodel_arch* perf_arch)
{
	double speedup = 0;
	int dev;
	for(dev = 0; dev < perf_arch->ndevices; dev++)
	{
		enum starpu_worker_archtype archtype = perf_arch->devices[dev].type;
		double coef = starpu_driver_info[archtype].alpha;
		speedup += coef * (perf_arch->devices[dev].ncores);
	}
	return speedup;
}

static double common_task_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, struct starpu_task *task, unsigned nimpl)
{
	double exp;
	double alpha;

	STARPU_ASSERT_MSG(model->cost_function, "STARPU_COMMON requires common cost_function to be defined");

	exp = model->cost_function(task, nimpl);
	alpha = starpu_worker_get_relative_speedup(arch);

	STARPU_ASSERT(!_STARPU_IS_ZERO(alpha));

	return exp/alpha;
}

void _starpu_init_and_load_perfmodel(struct starpu_perfmodel *model)
{
	if (!model || model->is_loaded)
		return;

	starpu_perfmodel_init(model);

	if (model->is_loaded)
		return;

	switch (model->type)
	{
		case STARPU_PER_WORKER:
		case STARPU_PER_ARCH:
		case STARPU_COMMON:
			/* Nothing more to do than init */
			break;
		case STARPU_HISTORY_BASED:
		case STARPU_NL_REGRESSION_BASED:
			_starpu_load_history_based_model(model, 1);
			break;
		case STARPU_REGRESSION_BASED:
		case STARPU_MULTIPLE_REGRESSION_BASED:
			_starpu_load_history_based_model(model, 0);
			break;

		default:
			STARPU_ABORT();
	}

	model->is_loaded = 1;
}

static double starpu_model_expected_perf(struct starpu_task *task, struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch,  unsigned nimpl)
{
	double exp_perf = 0.0;
	if (model)
	{
		_starpu_init_and_load_perfmodel(model);

		struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

		switch (model->type)
		{
			case STARPU_PER_ARCH:
				exp_perf = per_arch_task_expected_perf(model, arch, task, nimpl);
				STARPU_ASSERT_MSG(isnan(exp_perf)||exp_perf>=0,"exp_perf=%lf\n",exp_perf);
				break;
			case STARPU_COMMON:
				exp_perf = common_task_expected_perf(model, arch, task, nimpl);
				STARPU_ASSERT_MSG(isnan(exp_perf)||exp_perf>=0,"exp_perf=%lf\n",exp_perf);
				break;
			case STARPU_HISTORY_BASED:
				exp_perf = _starpu_history_based_job_expected_perf(model, arch, j, nimpl);
				STARPU_ASSERT_MSG(isnan(exp_perf)||exp_perf>=0,"exp_perf=%lf\n",exp_perf);
				break;
			case STARPU_REGRESSION_BASED:
				exp_perf = _starpu_regression_based_job_expected_perf(model, arch, j, nimpl);
				STARPU_ASSERT_MSG(isnan(exp_perf)||exp_perf>=0,"exp_perf=%lf\n",exp_perf);
				break;
			case STARPU_NL_REGRESSION_BASED:
				exp_perf = _starpu_non_linear_regression_based_job_expected_perf(model, arch, j,nimpl);
				STARPU_ASSERT_MSG(isnan(exp_perf)||exp_perf>=0,"exp_perf=%lf\n",exp_perf);
				break;
			case STARPU_MULTIPLE_REGRESSION_BASED:
				exp_perf = _starpu_multiple_regression_based_job_expected_perf(model, arch, j, nimpl);
				STARPU_ASSERT_MSG(isnan(exp_perf)||exp_perf>=0,"exp_perf=%lf\n",exp_perf);
				break;
			default:
				STARPU_ABORT();
		}
	}

	/* no model was found */
	return exp_perf;
}

static double starpu_model_worker_expected_perf(struct starpu_task *task, struct starpu_perfmodel *model, unsigned workerid, unsigned sched_ctx_id, unsigned nimpl)
{
	if (!model)
		return 0.0;

	if (model->type == STARPU_PER_WORKER)
		return per_worker_task_expected_perf(model, workerid, task, nimpl);
	else
	{
		struct starpu_perfmodel_arch *per_arch = starpu_worker_get_perf_archtype(workerid, sched_ctx_id);
		return starpu_model_expected_perf(task, model, per_arch, nimpl);
	}
}

double starpu_task_expected_length(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;
	return starpu_model_expected_perf(task, task->cl->model, arch, nimpl);
}

double starpu_task_worker_expected_length(struct starpu_task *task, unsigned workerid, unsigned sched_ctx_id, unsigned nimpl)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;
	return starpu_model_worker_expected_perf(task, task->cl->model, workerid, sched_ctx_id, nimpl);
}

double starpu_task_expected_length_average(struct starpu_task *task, unsigned sched_ctx_id)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	double harmsum = 0.0;
	unsigned n = 0;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned nimpl;
		unsigned impl_mask;
		unsigned workerid = workers->get_next(workers, &it);

		if (!starpu_worker_can_execute_task_impl(workerid, task, &impl_mask))
			continue;

		double best_expected = DBL_MAX;
		for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!(impl_mask & (1U << nimpl)))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			double expected = starpu_task_worker_expected_length(task, workerid, sched_ctx_id, nimpl);
			if (expected < best_expected)
				best_expected = expected;
		}
		harmsum += 1. / best_expected;
		n++;
	}

	return n/harmsum;
}

double starpu_task_expected_energy(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;
	return starpu_model_expected_perf(task, task->cl->energy_model, arch, nimpl);
}

double starpu_task_worker_expected_energy(struct starpu_task *task, unsigned workerid, unsigned sched_ctx_id, unsigned nimpl)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;
	return starpu_model_worker_expected_perf(task, task->cl->energy_model, workerid, sched_ctx_id, nimpl);

}

double starpu_task_expected_energy_average(struct starpu_task *task, unsigned sched_ctx_id)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	double harmsum = 0.0;
	unsigned n = 0;

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned nimpl;
		unsigned impl_mask;
		unsigned workerid = workers->get_next(workers, &it);

		if (!starpu_worker_can_execute_task_impl(workerid, task, &impl_mask))
			continue;

		double best_expected = DBL_MAX;
		for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!(impl_mask & (1U << nimpl)))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			double expected = starpu_task_worker_expected_energy(task, workerid, sched_ctx_id, nimpl);
			if (expected < best_expected)
				best_expected = expected;
		}
		harmsum += 1. / best_expected;
		n++;
	}

	return n/harmsum;
}

double starpu_task_expected_conversion_time(struct starpu_task *task,
					    struct starpu_perfmodel_arch* arch,
					    unsigned nimpl)
{
	unsigned i;
	double sum = 0.0;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);

#ifdef STARPU_DEVEL
#warning TODO: conversion time with combined arch perfmodel
#endif
	STARPU_ASSERT_MSG(arch->ndevices == 1, "TODO");

	for (i = 0; i < nbuffers; i++)
	{
		starpu_data_handle_t handle;
		struct starpu_task *conversion_task;
		enum starpu_node_kind node_kind;

		handle = STARPU_TASK_GET_HANDLE(task, i);
		if (!_starpu_data_is_multiformat_handle(handle))
			continue;

		node_kind = starpu_worker_get_memory_node_kind(arch->devices[0].type);
		if (!_starpu_handle_needs_conversion_task_for_arch(handle, node_kind))
			continue;

		conversion_task = _starpu_create_conversion_task_for_arch(handle, node_kind);
		sum += starpu_task_expected_length(conversion_task, arch, nimpl);
		_starpu_spin_lock(&handle->header_lock);
		handle->refcnt--;
		handle->busy_count--;
		if (!_starpu_data_check_not_busy(handle))
			_starpu_spin_unlock(&handle->header_lock);
		starpu_task_clean(conversion_task);
		free(conversion_task);
	}

	return sum;
}

/* Predict the transfer time (in µs) to move a handle between memory nodes */
static double _starpu_data_expected_transfer_time(starpu_data_handle_t handle, unsigned src_node, unsigned dst_node, enum starpu_data_access_mode mode, size_t size)
{
	double duration = 0.;
#define MAX_REQUESTS 4
	unsigned src_nodes[MAX_REQUESTS];
	unsigned dst_nodes[MAX_REQUESTS];
	unsigned handling_nodes[MAX_REQUESTS];
	int nhops = _starpu_determine_request_path(handle, src_node, dst_node, mode,
			MAX_REQUESTS,
			src_nodes, dst_nodes, handling_nodes, 0);
	int i;

	for (i = 0; i < nhops; i++)
		duration += starpu_transfer_predict(src_nodes[i], dst_nodes[i], size);

	return duration;
}

/* Predict the transfer time (in µs) to move a handle to a memory node */
double starpu_data_expected_transfer_time(starpu_data_handle_t handle, unsigned memory_node, enum starpu_data_access_mode mode)
{
	/* FIXME: Fix write-only mode with _starpu_expected_transfer_time_writeback */
	/* FIXME: count time_writeback only if the data is not dirty. Once it is dirty, we shouldn't
	 * count the writeback penatly again. */

	/* If we don't need to read the content of the handle */
	if (!(mode & STARPU_R))
		return 0.0;

	if (starpu_data_is_on_node(handle, memory_node))
		return 0.0;

	size_t size = _starpu_data_get_size(handle);

	/* XXX in case we have an abstract piece of data (eg.  with the
	 * void interface, this does not introduce any overhead, and we
	 * don't even want to consider the latency that is not
	 * relevant). */
	if (size == 0)
		return 0.0;

	double duration = 0.;

	int src_node = _starpu_select_src_node(handle, memory_node);
	if (src_node >= 0)
	{
		duration += _starpu_data_expected_transfer_time(handle, src_node, memory_node, mode, size);
	}
	/* Else, will just create it in place. Ideally we should take the
	 * time to create it into account */

	if (_starpu_expected_transfer_time_writeback && (mode & STARPU_W) && handle->home_node >= 0)
	{
		/* Will have to write back the produced data, artificially count
		 * the time to bring it back to its home node */
		duration += _starpu_data_expected_transfer_time(handle, memory_node, handle->home_node, STARPU_R, size);
	}

	return duration;
}

/* Data transfer performance modeling */
double starpu_task_expected_data_transfer_time(unsigned memory_node, struct starpu_task *task)
{
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	unsigned buffer;

	double penalty = 0.0;

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, buffer);
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, buffer);
		int node = _starpu_task_data_get_node_on_node(task, buffer, memory_node);

		if (node >= 0)
			penalty += starpu_data_expected_transfer_time(handle, node, mode);
	}

	return penalty;
}

/* Data transfer performance modeling */
double starpu_task_expected_data_transfer_time_for(struct starpu_task *task, unsigned worker)
{
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	unsigned buffer;

	double penalty = 0.0;

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, buffer);
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, buffer);
		int node = _starpu_task_data_get_node_on_worker(task, buffer, worker);

		if (node >= 0)
			penalty += starpu_data_expected_transfer_time(handle, node, mode);
	}

	return penalty;
}

/* Return the expected duration of the entire task bundle in µs */
double starpu_task_bundle_expected_length(starpu_task_bundle_t bundle, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	double expected_length = 0.0;

	/* We expect the length of the bundle the be the sum of the different tasks length. */
	STARPU_PTHREAD_MUTEX_LOCK(&bundle->mutex);

	struct _starpu_task_bundle_entry *entry;
	entry = bundle->list;

	while (entry)
	{
		if(!entry->task->scheduled)
		{
			double task_length = starpu_task_expected_length(entry->task, arch, nimpl);

			/* In case the task is not calibrated, we consider the task
			 * ends immediately. */
			if (task_length > 0.0)
				expected_length += task_length;
		}

		entry = entry->next;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	return expected_length;
}

/* Return the expected energy consumption of the entire task bundle in J */
double starpu_task_bundle_expected_energy(starpu_task_bundle_t bundle, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	double expected_energy = 0.0;

	/* We expect total consumption of the bundle the be the sum of the different tasks consumption. */
	STARPU_PTHREAD_MUTEX_LOCK(&bundle->mutex);

	struct _starpu_task_bundle_entry *entry;
	entry = bundle->list;

	while (entry)
	{
		double task_energy = starpu_task_expected_energy(entry->task, arch, nimpl);

		/* In case the task is not calibrated, we consider the task
		 * ends immediately. */
		if (task_energy > 0.0)
			expected_energy += task_energy;

		entry = entry->next;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	return expected_energy;
}

/* Return the time (in µs) expected to transfer all data used within the bundle */
double starpu_task_bundle_expected_data_transfer_time(starpu_task_bundle_t bundle, unsigned memory_node)
{
	STARPU_PTHREAD_MUTEX_LOCK(&bundle->mutex);

	struct _starpu_handle_list *handles = NULL;

	/* We list all the handle that are accessed within the bundle. */

	/* For each task in the bundle */
	struct _starpu_task_bundle_entry *entry = bundle->list;
	while (entry)
	{
		struct starpu_task *task = entry->task;

		if (task->cl)
		{
			unsigned b;
			unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
			for (b = 0; b < nbuffers; b++)
			{
				starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, b);
				enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, b);

				if (!(mode & STARPU_R))
					continue;

				/* Insert the handle in the sorted list in case
				 * it's not already in that list. */
				_starpu_insertion_handle_sorted(&handles, handle, mode);
			}
		}

		entry = entry->next;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	/* Compute the sum of data transfer time, and destroy the list */

	double total_exp = 0.0;

	while (handles)
	{
		struct _starpu_handle_list *current = handles;
		handles = handles->next;

		double exp;
		exp = starpu_data_expected_transfer_time(current->handle, memory_node, current->mode);

		total_exp += exp;

		free(current);
	}

	return total_exp;
}

#define _PERF_MODEL_DIR_MAXLEN 256
#define _PERF_MODEL_DIR_MAXNB  20

static char *_perf_model_paths[_PERF_MODEL_DIR_MAXNB];
static int _perf_model_paths_nb=0;

static int _perf_model_bus_location = -1;
static int _perf_model_bus_directory_existence_was_tested[_PERF_MODEL_DIR_MAXNB];
static char *_perf_model_dir_bus = NULL;
static char *_perf_model_dirs_codelet[_PERF_MODEL_DIR_MAXNB];

static int _perf_model_codelet_directory_existence_was_tested[_PERF_MODEL_DIR_MAXNB];

static void _starpu_set_perf_model_dirs();

void _starpu_find_perf_model_codelet(const char *symbol, const char *hostname, char *path, size_t maxlen)
{
	const char *dot = strrchr(symbol, '.');
	int i=0;

	_starpu_set_perf_model_dirs();

	for(i=0 ; _perf_model_paths[i]!=NULL ; i++)
	{
		snprintf(path, maxlen, "%scodelets/%d/%s%s%s", _perf_model_paths[i], _STARPU_PERFMODEL_VERSION, symbol, dot?"":".", dot?"":hostname);
		//_STARPU_MSG("checking file %s\n", path);
		int res = access(path, F_OK);
		if (res == 0)
		{
			return;
		}
	}

	// The file was not found
	path[0] = '\0';
}

void _starpu_find_perf_model_codelet_debug(const char *symbol, const char *hostname, const char *arch, char *path, size_t maxlen)
{
	const char *dot = strrchr(symbol, '.');
	int i=0;

	_starpu_set_perf_model_dirs();

	for(i=0 ; _perf_model_paths[i]!=NULL ; i++)
	{
		snprintf(path, maxlen, "%scodelets/%d/%s%s%s", _perf_model_paths[i], _STARPU_PERFMODEL_VERSION, symbol, dot?"":".", dot?"":hostname);
		//_STARPU_MSG("checking file %s\n", path);
		int res = access(path, F_OK);
		if (res == 0)
		{
			snprintf(path, maxlen, "%sdebug/%s%s%s%s", _perf_model_paths[i], symbol, dot?"":".", dot?"":hostname, arch);
			return;
		}
	}

	// The file was not found
	path[0] = '\0';
}

void _starpu_set_default_perf_model_codelet(const char *symbol, const char *hostname, char *path, size_t maxlen)
{
	_starpu_create_codelet_sampling_directory_if_needed(0);
	const char *dot = strrchr(symbol, '.');
	snprintf(path, maxlen, "%scodelets/%d/%s%s%s", _perf_model_paths[0], _STARPU_PERFMODEL_VERSION, symbol, dot?"":".", dot?"":hostname);
}

char *_starpu_get_perf_model_dir_default()
{
	_starpu_create_codelet_sampling_directory_if_needed(0);
	return _perf_model_paths[0];
}

char *_starpu_get_perf_model_dir_bus()
{
	int loc = _starpu_create_bus_sampling_directory_if_needed(-1);
	if (loc == -ENOENT)
		return NULL;
	if (_perf_model_dir_bus == NULL)
	{
		_STARPU_MALLOC(_perf_model_dir_bus, _PERF_MODEL_DIR_MAXLEN);
		snprintf(_perf_model_dir_bus, _PERF_MODEL_DIR_MAXLEN, "%sbus/", _perf_model_paths[_perf_model_bus_location]);
	}
	return _perf_model_dir_bus;
}

char **_starpu_get_perf_model_dirs_codelet()
{
	if (_perf_model_dirs_codelet[0] == NULL)
	{
		int i;
		for(i=0 ; i<_perf_model_paths_nb ; i++)
		{
			_STARPU_MALLOC(_perf_model_dirs_codelet[i], _PERF_MODEL_DIR_MAXLEN);
			snprintf(_perf_model_dirs_codelet[i], _PERF_MODEL_DIR_MAXLEN, "%scodelets/%d/", _perf_model_paths[i], _STARPU_PERFMODEL_VERSION);
			_starpu_create_codelet_sampling_directory_if_needed(i);
		}
	}
	return _perf_model_dirs_codelet;
}

static void _perf_model_add_dir(char *dir, int only_is_valid, char *var)
{
	STARPU_ASSERT_MSG(_perf_model_paths_nb < _PERF_MODEL_DIR_MAXNB, "Maximum number of performance models directory");

	if (dir == NULL || strlen(dir) == 0)
	{
		_STARPU_MSG("Warning: directory <%s> as set %s is empty\n", dir, var);
		return;
	}
	int add=1;
	if (only_is_valid)
	{
		DIR *ddir = opendir(dir);
		if (ddir == NULL)
		{
			add = 0;
			_STARPU_MSG("Warning: directory <%s> as set %s does not exist\n", dir, var);
		}
		else
			closedir(ddir);
	}

	if (add == 1)
	{
		_STARPU_DEBUG("Adding directory <%s> as set %s at location %d\n", dir, var, _perf_model_paths_nb);
		_STARPU_MALLOC(_perf_model_paths[_perf_model_paths_nb], _PERF_MODEL_DIR_MAXLEN);
		snprintf(_perf_model_paths[_perf_model_paths_nb], _PERF_MODEL_DIR_MAXLEN, "%s/", dir);
		_perf_model_bus_directory_existence_was_tested[_perf_model_paths_nb] = 0;
		_perf_model_codelet_directory_existence_was_tested[_perf_model_paths_nb] = 0;
		_perf_model_paths_nb ++;
		_perf_model_paths[_perf_model_paths_nb] = NULL;
	}
}

void _starpu_set_perf_model_dirs()
{
	if (_perf_model_paths_nb != 0) return;

	char *env = starpu_getenv("STARPU_PERF_MODEL_DIR");
	if (env)
	{
		_perf_model_add_dir(env, 0, "by variable STARPU_PERF_MODEL_DIR");
	}

#ifdef STARPU_PERF_MODEL_DIR
	_perf_model_add_dir((char *)STARPU_PERF_MODEL_DIR, 0, "by configure parameter");
#else
	char home[_PERF_MODEL_DIR_MAXLEN];
	snprintf(home, _PERF_MODEL_DIR_MAXLEN, "%s/.starpu/sampling", _starpu_get_home_path());
	_perf_model_add_dir(home, 0, "by STARPU_HOME directory");
#endif

	env = starpu_getenv("STARPU_PERF_MODEL_PATH");
	if (env)
	{
		char *saveptr, *token;
		token = strtok_r(env, ":", &saveptr);
		for (; token != NULL; token = strtok_r(NULL, ",", &saveptr))
		{
			_perf_model_add_dir(token, 1, "by variable STARPU_PERF_MODEL_PATH");
		}
	}

	_perf_model_add_dir(_STARPU_STRINGIFY(STARPU_SAMPLING_DIR), 1, "by installation directory");
}

int _starpu_set_default_perf_model_bus()
{
	assert(_perf_model_bus_location < 0);
	_perf_model_bus_location = 0;
	return _perf_model_bus_location;
}

int _starpu_get_perf_model_bus()
{
	if (_perf_model_bus_location != -1)
		return _perf_model_bus_location;

	char hostname[65];
	int i=0;

	_starpu_set_perf_model_dirs();
	_starpu_gethostname(hostname, sizeof(hostname));

	while(_perf_model_paths[i])
	{
		char path[PATH_LENGTH];
		snprintf(path, PATH_LENGTH, "%sbus/%s.config", _perf_model_paths[i], hostname);
		_STARPU_DEBUG("checking path %s\n", path);
		int res = access(path, F_OK);
		if (res == 0)
		{
			_perf_model_bus_location = i;
			return _perf_model_bus_location;
		}
		i++;
	}
	return -ENOENT;
}

int _starpu_create_bus_sampling_directory_if_needed(int location)
{
	if (location < 0)
		location = _starpu_get_perf_model_bus();
	if (location == -ENOENT)
		return -ENOENT;

	STARPU_ASSERT_MSG(location < _perf_model_paths_nb, "Location %d for performance models file is invalid", location);
	if (!_perf_model_bus_directory_existence_was_tested[location])
	{
		char *dir = _perf_model_paths[location];

		_STARPU_DEBUG("creating directories at <%s>\n", dir);

		/* The performance of the codelets are stored in
		 * $STARPU_PERF_MODEL_DIR/codelets/ while those of the bus are stored in
		 * $STARPU_PERF_MODEL_DIR/bus/ so that we don't have name collisions */

		_starpu_mkpath_and_check(dir, S_IRWXU);

		/* Performance of the memory subsystem */
		char bus[_PERF_MODEL_DIR_MAXLEN];
		snprintf(bus, _PERF_MODEL_DIR_MAXLEN, "%s/bus/", dir);
		_starpu_mkpath_and_check(bus, S_IRWXU);

		_perf_model_bus_directory_existence_was_tested[location] = 1;
	}
	return 0;
}

void _starpu_create_codelet_sampling_directory_if_needed(int location)
{
	STARPU_ASSERT_MSG(location < _perf_model_paths_nb, "Location %d for performance models file is invalid", location);
	if (!_perf_model_codelet_directory_existence_was_tested[location])
	{
		char *dir = _perf_model_paths[location];

		if (dir)
		{
			_STARPU_DEBUG("creating directories at <%s>\n", dir);

			/* Per-task performance models */
			char codelet[_PERF_MODEL_DIR_MAXLEN];
			snprintf(codelet, _PERF_MODEL_DIR_MAXLEN, "%scodelets/%d/", dir, _STARPU_PERFMODEL_VERSION);
			_starpu_mkpath_and_check(codelet, S_IRWXU);

			/* Performance debug measurements */
			char debug[_PERF_MODEL_DIR_MAXLEN];
			snprintf(debug, _PERF_MODEL_DIR_MAXLEN, "%sdebug/", dir);
			_starpu_mkpath(debug, S_IRWXU);

			_perf_model_codelet_directory_existence_was_tested[location] = 1;
		}
	}
}

void starpu_perfmodel_free_sampling(void)
{
	int i;
	for(i=0 ; i<_perf_model_paths_nb ; i++)
	{
		free(_perf_model_paths[i]);
		_perf_model_paths[i] = NULL;
		_perf_model_bus_directory_existence_was_tested[i] = 0;
		_perf_model_codelet_directory_existence_was_tested[i] = 0;
		free(_perf_model_dirs_codelet[i]);
		_perf_model_dirs_codelet[i] = NULL;
	}
	_perf_model_paths_nb = 0;
	_perf_model_bus_location = -1;
	free(_perf_model_dir_bus);
	_perf_model_dir_bus = NULL;
	_starpu_free_arch_combs();
}


static double nop_cost_function(struct starpu_task *t STARPU_ATTRIBUTE_UNUSED, struct starpu_perfmodel_arch *a STARPU_ATTRIBUTE_UNUSED, unsigned i STARPU_ATTRIBUTE_UNUSED)
{
	return 0.000001;
}

struct starpu_perfmodel starpu_perfmodel_nop =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = nop_cost_function,
};

/* This function is intended to be used by external tools that should read
 * the performance model files */
int starpu_perfmodel_list(FILE *output)
{
#ifdef HAVE_SCANDIR
	struct dirent **list;
	int i=0;

	_starpu_set_perf_model_dirs();

	for(i=0 ; _perf_model_paths[i]!=NULL ; i++)
	{
		char pcodelet[_PERF_MODEL_DIR_MAXLEN];
		int n;

		snprintf(pcodelet, _PERF_MODEL_DIR_MAXLEN, "%scodelets/%d/", _perf_model_paths[i], _STARPU_PERFMODEL_VERSION);
		n = scandir(pcodelet, &list, NULL, alphasort);
		if (n < 0)
		{
			_STARPU_DISP("Could not open the perfmodel directory <%s>: %s\n", pcodelet, strerror(errno));
		}
		else
		{
			int j;
			fprintf(output, "codelet directory: <%s>\n", pcodelet);
			for (j = 0; j < n; j++)
			{
				if (strcmp(list[j]->d_name, ".") && strcmp(list[j]->d_name, ".."))
					fprintf(output, "file: <%s>\n", list[j]->d_name);
				free(list[j]);
			}
			free(list);
		}
	}
	return 0;
#else
	(void)output;
	_STARPU_MSG("Listing perfmodels is not implemented on pure Windows yet\n");
	return 1;
#endif
}

void starpu_perfmodel_directory(FILE *output)
{
	int i;

	_starpu_set_perf_model_dirs();

	for(i=0 ; _perf_model_paths[i]!=NULL ; i++)
	{
		char pcodelet[_PERF_MODEL_DIR_MAXLEN];
		snprintf(pcodelet, _PERF_MODEL_DIR_MAXLEN, "%scodelets/%d/", _perf_model_paths[i], _STARPU_PERFMODEL_VERSION);
		fprintf(output, "directory: <%s>\n", pcodelet);
	}
}
