/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <unistd.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>

/*
 * PER ARCH model
 */

static double per_arch_job_expected_length(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct job_s *j)
{
	double exp = -1.0;
	double (*per_arch_cost_model)(struct starpu_buffer_descr_t *);
	
	if (!model->is_loaded)
	{
		if (starpu_get_env_number("CALIBRATE") != -1)
		{
			fprintf(stderr, "CALIBRATE model %s\n", model->symbol);
			model->benchmarking = 1;
		}
		else {
			model->benchmarking = 0;
		}
		
		register_model(model);
		model->is_loaded = 1;
	}

	per_arch_cost_model = model->per_arch[arch].cost_model;

	if (per_arch_cost_model)
		exp = per_arch_cost_model(j->task->buffers);

	return exp;
}

/*
 * Common model
 */

static double common_job_expected_length(struct starpu_perfmodel_t *model, uint32_t who, struct job_s *j)
{
	double exp;

	if (model->cost_model) {
		float alpha;
		exp = model->cost_model(j->task->buffers);
		switch (who) {
			case CORE:
				alpha = CORE_ALPHA;
				break;
			case CUDA:
				alpha = CUDA_ALPHA;
				break;
			default:
				/* perhaps there are various worker types on that queue */
				alpha = 1.0; // this value is not significant ...
				break;
		}

		STARPU_ASSERT(alpha != 0.0f);

		return (exp/alpha);
	}

	return -1.0;
}

double job_expected_length(uint32_t who, struct job_s *j, enum starpu_perf_archtype arch)
{
	struct starpu_perfmodel_t *model = j->task->cl->model;

	if (model) {
		switch (model->type) {
			case PER_ARCH:
				return per_arch_job_expected_length(model, arch, j);

			case COMMON:
				return common_job_expected_length(model, who, j);

			case HISTORY_BASED:
				return history_based_job_expected_length(model, arch, j);

			case REGRESSION_BASED:
				return regression_based_job_expected_length(model, arch, j);

			default:
				STARPU_ASSERT(0);
		};
	}

	/* no model was found */
	return 0.0;
}


/* Data transfer performance modeling */
double data_expected_penalty(struct jobq_s *q, struct job_s *j)
{
	uint32_t memory_node = q->memory_node;
	unsigned nbuffers = j->task->cl->nbuffers;
	unsigned buffer;

	double penalty = 0.0;

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		data_state *state = j->task->buffers[buffer].handle;

		if (!is_data_present_or_requested(state, memory_node))
		{
			/* TODO */
			penalty += 1000.0;
		}
	}

	return penalty;
}

