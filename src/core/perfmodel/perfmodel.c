/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

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
		double coef = 0.0;
		if (perf_arch->devices[dev].type == STARPU_CPU_WORKER)
			coef = _STARPU_CPU_ALPHA;
		else if (perf_arch->devices[dev].type == STARPU_CUDA_WORKER)
			coef = _STARPU_CUDA_ALPHA;
		else if (perf_arch->devices[dev].type == STARPU_OPENCL_WORKER)
			coef = _STARPU_OPENCL_ALPHA;
		else if (perf_arch->devices[dev].type == STARPU_MIC_WORKER)
			coef = _STARPU_MIC_ALPHA;
		else if (perf_arch->devices[dev].type == STARPU_MPI_MS_WORKER)
			coef = _STARPU_MPI_MS_ALPHA;

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

double starpu_task_expected_length(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;
	return starpu_model_expected_perf(task, task->cl->model, arch, nimpl);
}

double starpu_task_expected_energy(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl)
{
	if (!task->cl)
		/* Tasks without codelet don't actually take time */
		return 0.0;
	return starpu_model_expected_perf(task, task->cl->energy_model, arch, nimpl);
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

		node_kind = _starpu_worker_get_node_kind(arch->devices[0].type);
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

/* Predict the transfer time (in µs) to move a handle to a memory node */
double starpu_data_expected_transfer_time(starpu_data_handle_t handle, unsigned memory_node, enum starpu_data_access_mode mode)
{
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

	int src_node = _starpu_select_src_node(handle, memory_node);
	if (src_node < 0)
		/* Will just create it in place. Ideally we should take the
		 * time to create it into account */
		return 0.0;

#define MAX_REQUESTS 4
	unsigned src_nodes[MAX_REQUESTS];
	unsigned dst_nodes[MAX_REQUESTS];
	unsigned handling_nodes[MAX_REQUESTS];
	int nhops = _starpu_determine_request_path(handle, src_node, memory_node, mode,
			MAX_REQUESTS,
			src_nodes, dst_nodes, handling_nodes, 0);
	int i;
	double duration = 0.;

	for (i = 0; i < nhops; i++)
		duration += starpu_transfer_predict(src_nodes[i], dst_nodes[i], size);
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
				_insertion_handle_sorted(&handles, handle, mode);
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

static int directory_existence_was_tested = 0;
static char *_perf_model_dir = NULL;
static char *_perf_model_dir_codelet = NULL;
static char *_perf_model_dir_bus = NULL;
static char *_perf_model_dir_debug = NULL;
#define _PERF_MODEL_DIR_MAXLEN 256

void _starpu_set_perf_model_dirs()
{
	_STARPU_MALLOC(_perf_model_dir, _PERF_MODEL_DIR_MAXLEN);
	_STARPU_MALLOC(_perf_model_dir_codelet, _PERF_MODEL_DIR_MAXLEN);
	_STARPU_MALLOC(_perf_model_dir_bus, _PERF_MODEL_DIR_MAXLEN);
	_STARPU_MALLOC(_perf_model_dir_debug, _PERF_MODEL_DIR_MAXLEN);

#ifdef STARPU_PERF_MODEL_DIR
	/* use the directory specified at configure time */
	snprintf(_perf_model_dir, _PERF_MODEL_DIR_MAXLEN, "%s", (char *)STARPU_PERF_MODEL_DIR);
#else
	snprintf(_perf_model_dir, _PERF_MODEL_DIR_MAXLEN, "%s/.starpu/sampling/", _starpu_get_home_path());
#endif
	char *path = starpu_getenv("STARPU_PERF_MODEL_DIR");
	if (path)
	{
		snprintf(_perf_model_dir, _PERF_MODEL_DIR_MAXLEN, "%s/", path);
	}

	snprintf(_perf_model_dir_codelet, _PERF_MODEL_DIR_MAXLEN, "%s/codelets/%d/", _perf_model_dir, _STARPU_PERFMODEL_VERSION);
	snprintf(_perf_model_dir_bus, _PERF_MODEL_DIR_MAXLEN, "%s/bus/", _perf_model_dir);
	snprintf(_perf_model_dir_debug, _PERF_MODEL_DIR_MAXLEN, "%s/debug/", _perf_model_dir);
}

char *_starpu_get_perf_model_dir_codelet()
{
	_starpu_create_sampling_directory_if_needed();
	return _perf_model_dir_codelet;
}

char *_starpu_get_perf_model_dir_bus()
{
	_starpu_create_sampling_directory_if_needed();
	return _perf_model_dir_bus;
}

char *_starpu_get_perf_model_dir_debug()
{
	_starpu_create_sampling_directory_if_needed();
	return _perf_model_dir_debug;
}

void _starpu_create_sampling_directory_if_needed(void)
{
	if (!directory_existence_was_tested)
	{
		_starpu_set_perf_model_dirs();

		/* The performance of the codelets are stored in
		 * $STARPU_PERF_MODEL_DIR/codelets/ while those of the bus are stored in
		 * $STARPU_PERF_MODEL_DIR/bus/ so that we don't have name collisions */

		/* Testing if a directory exists and creating it otherwise
		   may not be safe: it is possible that the permission are
		   changed in between. Instead, we create it and check if
		   it already existed before */
		_starpu_mkpath_and_check(_perf_model_dir, S_IRWXU);

		/* Per-task performance models */
		_starpu_mkpath_and_check(_perf_model_dir_codelet, S_IRWXU);

		/* Performance of the memory subsystem */
		_starpu_mkpath_and_check(_perf_model_dir_bus, S_IRWXU);

		/* Performance debug measurements */
		_starpu_mkpath(_perf_model_dir_debug, S_IRWXU);

		directory_existence_was_tested = 1;
	}
}

void starpu_perfmodel_free_sampling(void)
{
	free(_perf_model_dir);
	_perf_model_dir = NULL;
	free(_perf_model_dir_codelet);
	_perf_model_dir_codelet = NULL;
	free(_perf_model_dir_bus);
	_perf_model_dir_bus = NULL;
	free(_perf_model_dir_debug);
	_perf_model_dir_debug = NULL;
	directory_existence_was_tested = 0;
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

