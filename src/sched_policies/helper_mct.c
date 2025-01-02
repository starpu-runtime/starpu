/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Simon Archipoff
 * Copyright (C) 2020-2020  Télécom Sud Paris
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

#include <starpu_sched_component.h>
#include "helper_mct.h"
#include <float.h>

/* Alpha, Beta and Gamma are MCT-specific values, which allows the
 * user to set more precisely the weight of each computing value.
 * Beta, for example, controls the weight of communications between
 * memories for the computation of the best component to choose.
 */
#define _STARPU_SCHED_ALPHA_DEFAULT 1.0
#define _STARPU_SCHED_BETA_DEFAULT 1.0
#define _STARPU_SCHED_GAMMA_DEFAULT 1000.0

struct _starpu_mct_data *starpu_mct_init_parameters(struct starpu_sched_component_mct_data *params)
{
	struct _starpu_mct_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	if (params)
	{
		data->alpha = params->alpha;
		data->beta = params->beta;
		/* data->_gamma: cost of one Joule in us. If gamma is set to 10^6, then one Joule cost 1s */
		data->_gamma = params->_gamma;
		/* data->idle_power: Idle power of the whole machine in Watt */
		data->idle_power = params->idle_power;
	}
	else
	{
		data->alpha = starpu_getenv_float_default("STARPU_SCHED_ALPHA", _STARPU_SCHED_ALPHA_DEFAULT);
		data->beta = starpu_getenv_float_default("STARPU_SCHED_BETA", _STARPU_SCHED_BETA_DEFAULT);
#ifdef STARPU_NON_BLOCKING_DRIVERS
		if (starpu_getenv("STARPU_SCHED_GAMMA"))
			_STARPU_DISP("Warning: STARPU_SCHED_GAMMA was used, but --enable-blocking-drivers configuration was not set, CPU cores will not actually be sleeping\n");
#endif
		data->_gamma = starpu_getenv_float_default("STARPU_SCHED_GAMMA", _STARPU_SCHED_GAMMA_DEFAULT);
		data->idle_power = starpu_getenv_float_default("STARPU_IDLE_POWER", 0.0);
	}

	return data;
}

/* compute predicted_end by taking into account the case of the predicted transfer and the predicted_end overlap
 */
static double compute_expected_time(double now, double predicted_end, double predicted_length, double predicted_transfer)
{
	STARPU_ASSERT(!isnan(now + predicted_end + predicted_length + predicted_transfer));
	STARPU_ASSERT_MSG(now >= 0.0 && predicted_end >= 0.0 && predicted_length >= 0.0 && predicted_transfer >= 0.0, "now=%lf, predicted_end=%lf, predicted_length=%lf, predicted_transfer=%lf\n", now, predicted_end, predicted_length, predicted_transfer);

	/* TODO: actually schedule transfers */
	/* Compute the transfer time which will not be overlapped */
	/* However, no modification in calling function so that the whole transfer time is counted as a penalty */
	if (now + predicted_transfer < predicted_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer -= (predicted_end - now);
	}

	predicted_end += predicted_transfer;
	predicted_end += predicted_length;

	return predicted_end;
}

double starpu_mct_compute_fitness(struct _starpu_mct_data * d, double exp_end, double min_exp_end_of_task, double max_exp_end_of_workers, double transfer_len, double local_energy)
{
	if(isnan(local_energy))
		/* Energy not calibrated yet, but we cannot do this
		 * automatically anyway, so ignoring this for now */
		local_energy = 0.;

	/* Note: the expected end includes the data transfer duration, which we want to be able to tune separately */

	/* min_exp_end_of_task is the minimum end time of the task over all workers */
	double fitness = d->alpha * (exp_end - min_exp_end_of_task) + d->beta * transfer_len + d->_gamma * local_energy;

	/* max_exp_end is the maximum end time of the workers. If the total execution time is increased, then an
	  additional energy penalty must be considered*/
	if(exp_end > max_exp_end_of_workers)
		fitness += d->_gamma * d->idle_power * (exp_end - max_exp_end_of_workers) / 1000000.0; /* Since gamma is the cost in us of one Joules,
											       then  d->idle_power * (exp_end - max_exp_end)
											       must be in Joules, thus the / 1000000.0 */

	return fitness;
}

unsigned starpu_mct_compute_execution_times(struct starpu_sched_component *component, struct starpu_task *task,
				       double *estimated_lengths, double *estimated_transfer_length, unsigned *suitable_components)
{
	unsigned nsuitable_components = 0;

	unsigned i;
	for(i = 0; i < component->nchildren; i++)
	{
		struct starpu_sched_component * c = component->children[i];

		/* Silence static analysis warnings */
		estimated_lengths[i] = NAN;
		estimated_transfer_length[i] = NAN;

		if(starpu_sched_component_execute_preds(c, task, estimated_lengths + i))
		{
			if(isnan(estimated_lengths[i]))
				/* The perfmodel had been purged since the task was pushed
				 * onto the mct component. */
				continue;
			STARPU_ASSERT_MSG(estimated_lengths[i]>=0, "component=%p, child[%u]=%p, estimated_lengths[%u]=%lf\n", component, i, c, i, estimated_lengths[i]);

			estimated_transfer_length[i] = starpu_sched_component_transfer_length(c, task);
			suitable_components[nsuitable_components++] = i;
		}
	}
	return nsuitable_components;
}

void starpu_mct_compute_expected_times(struct starpu_sched_component *component, struct starpu_task *task STARPU_ATTRIBUTE_UNUSED,
		double *estimated_lengths, double *estimated_transfer_length, double *estimated_ends_with_task,
				       double *min_exp_end_of_task, double *max_exp_end_of_workers, unsigned *suitable_components, unsigned nsuitable_components)
{
	unsigned i;
	double now = starpu_timing_now();
	*min_exp_end_of_task = DBL_MAX;
	*max_exp_end_of_workers = 0.0;
	for(i = 0; i < nsuitable_components; i++)
	{
		unsigned icomponent = suitable_components[i];
		struct starpu_sched_component * c = component->children[icomponent];
		/* Estimated availability of worker */
		double estimated_end = c->estimated_end(c);
		if (estimated_end < now)
			estimated_end = now;
		estimated_ends_with_task[icomponent] = compute_expected_time(now,
								    estimated_end,
								    estimated_lengths[icomponent],
								    estimated_transfer_length[icomponent]);

		/* estimated_ends_with_task[icomponent]: estimated end of execution on the worker icomponent
		   estimated_end: estimatated end of the worker
		   min_exp_end_of_task: minimum estimated execution time of the task over all workers
		   max_exp_end_of_workers: maximum estimated end of the already-scheduled tasks over all workers
		*/
		if(estimated_ends_with_task[icomponent] < *min_exp_end_of_task)
			*min_exp_end_of_task = estimated_ends_with_task[icomponent];
		if(estimated_end > *max_exp_end_of_workers)
			*max_exp_end_of_workers = estimated_end;
	}
}

/* This function retrieves the energy consumption of a task in Joules*/
void starpu_mct_compute_energy(struct starpu_sched_component *component, struct starpu_task *task , double *local_energy, unsigned *suitable_components, unsigned nsuitable_components)
{
	unsigned i;
	for(i = 0; i < nsuitable_components; i++)
	{
		unsigned icomponent = suitable_components[i];
		int nimpl = 0;
		local_energy[icomponent] = starpu_task_worker_expected_energy(task, icomponent,  component->tree->sched_ctx_id, nimpl);
		for (nimpl  = 1; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			double e;
			e = starpu_task_worker_expected_energy(task, icomponent,  component->tree->sched_ctx_id, nimpl);
			if (e < local_energy[icomponent])
				local_energy[icomponent] = e;
		}
	}
}

int starpu_mct_get_best_component(struct _starpu_mct_data *d, struct starpu_task *task, double *estimated_lengths, double *estimated_transfer_length, double *estimated_ends_with_task, double *local_energy, double min_exp_end_of_task, double max_exp_end_of_workers, unsigned *suitable_components, unsigned nsuitable_components)
{
	double best_fitness = DBL_MAX;
	int best_icomponent = -1;
	unsigned i;

	for(i = 0; i < nsuitable_components; i++)
	{
		int icomponent = suitable_components[i];
		double tmp = starpu_mct_compute_fitness(d,
					     estimated_ends_with_task[icomponent],
					     min_exp_end_of_task,
					     max_exp_end_of_workers,
					     estimated_transfer_length[icomponent],
					     local_energy[icomponent]);

		if(tmp < best_fitness)
		{
			best_fitness = tmp;
			best_icomponent = icomponent;
		}
	}

	if (best_icomponent != -1)
	{
		task->predicted = estimated_lengths[best_icomponent];
		task->predicted_transfer = estimated_transfer_length[best_icomponent];
	}

	return best_icomponent;
}
