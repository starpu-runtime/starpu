/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <starpu_profiling.h>
#include <common/config.h>
#include <common/utils.h>
#include <unistd.h>
#include <sys/stat.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>

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

enum starpu_perf_archtype starpu_worker_get_perf_archtype(int workerid)
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	return config->workers[workerid].perf_arch;
}

/*
 * PER ARCH model
 */

static double per_arch_task_expected_length(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct starpu_task *task)
{
	double exp = -1.0;
	double (*per_arch_cost_model)(struct starpu_buffer_descr_t *);
	
	if (!model->is_loaded)
	{
		model->benchmarking = _starpu_get_calibrate_flag();
		
		_starpu_register_model(model);
		model->is_loaded = 1;
	}

	per_arch_cost_model = model->per_arch[arch].cost_model;

	if (per_arch_cost_model)
		exp = per_arch_cost_model(task->buffers);

	return exp;
}

/*
 * Common model
 */

double _starpu_worker_get_relative_speedup(int workerid)
{
	double alpha;
	enum starpu_archtype arch = starpu_worker_get_type(workerid);

	switch (arch) {
		case STARPU_CPU_WORKER:
			alpha = STARPU_CPU_ALPHA;
			break;
		case STARPU_CUDA_WORKER:
			alpha = STARPU_CUDA_ALPHA;
			break;
	        case STARPU_OPENCL_WORKER:
                        alpha = STARPU_OPENCL_ALPHA;
                        break;
		default:
			/* perhaps there are various worker types on that queue */
			alpha = 1.0; // this value is not significant ...
			break;
	}

	return alpha;
}

static double common_task_expected_length(struct starpu_perfmodel_t *model, int workerid, struct starpu_task *task)
{
	double exp;
	double alpha;

	if (model->cost_model) {
		exp = model->cost_model(task->buffers);
		alpha = _starpu_worker_get_relative_speedup(workerid);

		STARPU_ASSERT(alpha != 0.0f);

		return (exp/alpha);
	}

	return -1.0;
}

double _starpu_task_expected_length(int workerid, struct starpu_task *task, enum starpu_perf_archtype arch)
{
	starpu_job_t j = _starpu_get_job_associated_to_task(task);
	struct starpu_perfmodel_t *model = task->cl->model;

	if (model) {
		switch (model->type) {
			case STARPU_PER_ARCH:
				return per_arch_task_expected_length(model, arch, task);

			case STARPU_COMMON:
				return common_task_expected_length(model, workerid, task);

			case STARPU_HISTORY_BASED:
				return _starpu_history_based_job_expected_length(model, arch, j);

			case STARPU_REGRESSION_BASED:
				return _starpu_regression_based_job_expected_length(model, arch, j);

			default:
				STARPU_ABORT();
		};
	}

	/* no model was found */
	return 0.0;
}

/* Data transfer performance modeling */
double _starpu_data_expected_penalty(uint32_t memory_node, struct starpu_task *task)
{
	unsigned nbuffers = task->cl->nbuffers;
	unsigned buffer;

	double penalty = 0.0;

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_handle handle = task->buffers[buffer].handle;

		starpu_access_mode mode = task->buffers[buffer].mode;

		if ((mode == STARPU_W) || (mode == STARPU_SCRATCH))
			continue;

		if (!_starpu_is_data_present_or_requested(handle, memory_node))
		{
			size_t size = _starpu_data_get_size(handle);

			uint32_t src_node = _starpu_select_src_node(handle);

			penalty += _starpu_predict_transfer_time(src_node, memory_node, size);
		}
	}

	return penalty;
}

static int directory_existence_was_tested = 0;

void _starpu_get_perf_model_dir(char *path, size_t maxlen)
{
#ifdef STARPU_PERF_MODEL_DIR
	/* use the directory specified at configure time */
	snprintf(path, maxlen, "%s", STARPU_PERF_MODEL_DIR);
#else
	/* by default, we use $HOME/.starpu/sampling */
	const char *home_path = getenv("HOME");
	if (!home_path)
		home_path = getenv("USERPROFILE");
	if (!home_path) {
		_STARPU_ERROR("couldn't find a home place to put starpu data\n");
	}
	snprintf(path, maxlen, "%s/.starpu/sampling/", home_path);
#endif
}

void _starpu_get_perf_model_dir_codelets(char *path, size_t maxlen)
{
	_starpu_get_perf_model_dir(path, maxlen);
	strncat(path, "codelets/", maxlen);
}

void _starpu_get_perf_model_dir_bus(char *path, size_t maxlen)
{
	_starpu_get_perf_model_dir(path, maxlen);
	strncat(path, "bus/", maxlen);
}

void _starpu_get_perf_model_dir_debug(char *path, size_t maxlen)
{
	_starpu_get_perf_model_dir(path, maxlen);
	strncat(path, "debug/", maxlen);
}

void _starpu_create_sampling_directory_if_needed(void)
{
	if (!directory_existence_was_tested)
	{
		char perf_model_dir[256];
		_starpu_get_perf_model_dir(perf_model_dir, 256);

		/* The performance of the codelets are stored in
		 * $STARPU_PERF_MODEL_DIR/codelets/ while those of the bus are stored in
		 * $STARPU_PERF_MODEL_DIR/bus/ so that we don't have name collisions */
		
		/* Testing if a directory exists and creating it otherwise 
		   may not be safe: it is possible that the permission are
		   changed in between. Instead, we create it and check if
		   it already existed before */
		int ret;
		ret = _starpu_mkpath(perf_model_dir, S_IRWXU);

		if (ret == -1)
		{
			STARPU_ASSERT(errno == EEXIST);
	
			/* make sure that it is actually a directory */
			struct stat sb;
			stat(perf_model_dir, &sb);
			STARPU_ASSERT(S_ISDIR(sb.st_mode));
		}
	
		/* Per-task performance models */
		char perf_model_dir_codelets[256];
		_starpu_get_perf_model_dir_codelets(perf_model_dir_codelets, 256);

		ret = _starpu_mkpath(perf_model_dir_codelets, S_IRWXU);
		if (ret == -1)
		{
			STARPU_ASSERT(errno == EEXIST);
	
			/* make sure that it is actually a directory */
			struct stat sb;
			stat(perf_model_dir_codelets, &sb);
			STARPU_ASSERT(S_ISDIR(sb.st_mode));
		}
	
		/* Performance of the memory subsystem */
		char perf_model_dir_bus[256];
		_starpu_get_perf_model_dir_bus(perf_model_dir_bus, 256);

		ret = _starpu_mkpath(perf_model_dir_bus, S_IRWXU);
		if (ret == -1)
		{
			STARPU_ASSERT(errno == EEXIST);
	
			/* make sure that it is actually a directory */
			struct stat sb;
			stat(perf_model_dir_bus, &sb);
			STARPU_ASSERT(S_ISDIR(sb.st_mode));
		}
	
		/* Performance debug measurements */
		char perf_model_dir_debug[256];
		_starpu_get_perf_model_dir_debug(perf_model_dir_debug, 256);

		ret = _starpu_mkpath(perf_model_dir_debug, S_IRWXU);
		if (ret == -1)
		{
			STARPU_ASSERT(errno == EEXIST);
	
			/* make sure that it is actually a directory */
			struct stat sb;
			stat(perf_model_dir_debug, &sb);
			STARPU_ASSERT(S_ISDIR(sb.st_mode));
		}
	
		directory_existence_was_tested = 1;
	}
}
