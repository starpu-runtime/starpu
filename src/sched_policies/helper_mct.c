/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

/* Alpha, Beta and Gamma are MCT-specific values, which allows the
 * user to set more precisely the weight of each computing value.
 * Beta, for example, controls the weight of communications between
 * memories for the computation of the best component to choose. 
 */
#define _STARPU_SCHED_ALPHA_DEFAULT 1.0
#define _STARPU_SCHED_BETA_DEFAULT 1.0
#define _STARPU_SCHED_GAMMA_DEFAULT 1000.0

#ifdef STARPU_USE_TOP
static void param_modified(struct starpu_top_param* d)
{
	/* Just to show parameter modification. */
	_STARPU_MSG("%s has been modified : %f\n", d->name, *(double*) d->value);
}
#endif /* !STARPU_USE_TOP */

#ifdef STARPU_USE_TOP
static const float alpha_minimum=0;
static const float alpha_maximum=10.0;
static const float beta_minimum=0;
static const float beta_maximum=10.0;
static const float gamma_minimum=0;
static const float gamma_maximum=10000.0;
static const float idle_power_minimum=0;
static const float idle_power_maximum=10000.0;
#endif /* !STARPU_USE_TOP */

struct _starpu_mct_data *starpu_mct_init_parameters(struct starpu_sched_component_mct_data *params)
{
	struct _starpu_mct_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	if (params)
	{
		data->alpha = params->alpha;
		data->beta = params->beta;
		data->_gamma = params->_gamma;
		data->idle_power = params->idle_power;
	}
	else
	{
		data->alpha = starpu_get_env_float_default("STARPU_SCHED_ALPHA", _STARPU_SCHED_ALPHA_DEFAULT);
		data->beta = starpu_get_env_float_default("STARPU_SCHED_BETA", _STARPU_SCHED_BETA_DEFAULT);
		data->_gamma = starpu_get_env_float_default("STARPU_SCHED_GAMMA", _STARPU_SCHED_GAMMA_DEFAULT);
		data->idle_power = starpu_get_env_float_default("STARPU_IDLE_POWER", 0.0);
	}

#ifdef STARPU_USE_TOP
	starpu_top_register_parameter_float("MCT_ALPHA", &data->alpha,
					    alpha_minimum, alpha_maximum, param_modified);
	starpu_top_register_parameter_float("MCT_BETA", &data->beta,
					    beta_minimum, beta_maximum, param_modified);
	starpu_top_register_parameter_float("MCT_GAMMA", &data->_gamma,
					    gamma_minimum, gamma_maximum, param_modified);
	starpu_top_register_parameter_float("MCT_IDLE_POWER", &data->idle_power,
					    idle_power_minimum, idle_power_maximum, param_modified);
#endif /* !STARPU_USE_TOP */

	return data;
}

/* compute predicted_end by taking into account the case of the predicted transfer and the predicted_end overlap
 */
static double compute_expected_time(double now, double predicted_end, double predicted_length, double *predicted_transfer)
{
	STARPU_ASSERT(!isnan(now + predicted_end + predicted_length + *predicted_transfer));
	STARPU_ASSERT(now >= 0.0 && predicted_end >= 0.0 && predicted_length >= 0.0 && *predicted_transfer >= 0.0);

	/* TODO: actually schedule transfers */
	if (now + *predicted_transfer < predicted_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		*predicted_transfer = 0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		*predicted_transfer -= (predicted_end - now);
	}

	predicted_end += *predicted_transfer;
	predicted_end += predicted_length;

	return predicted_end;
}

double starpu_mct_compute_fitness(struct _starpu_mct_data * d, double exp_end, double min_exp_end, double max_exp_end, double transfer_len, double local_energy)
{
	/* Note: the expected end includes the data transfer duration, which we want to be able to tune separately */

	return d->alpha * (exp_end - min_exp_end)
		+ d->beta * transfer_len
		+ d->_gamma * local_energy
		+ d->_gamma * d->idle_power * (exp_end - max_exp_end);
}

int starpu_mct_compute_expected_times(struct starpu_sched_component *component, struct starpu_task *task,
		double *estimated_lengths, double *estimated_transfer_length, double *estimated_ends_with_task,
		double *min_exp_end_with_task, double *max_exp_end_with_task, int *suitable_components)
{
	int nsuitable_components = 0;

	int i;
	for(i = 0; i < component->nchildren; i++)
	{
		struct starpu_sched_component * c = component->children[i];
		if(starpu_sched_component_execute_preds(c, task, estimated_lengths + i))
		{
			if(isnan(estimated_lengths[i]))
				/* The perfmodel had been purged since the task was pushed
				 * onto the mct component. */
				continue;

			/* Estimated availability of worker */
			double estimated_end = c->estimated_end(c);
			double now = starpu_timing_now();
			if (estimated_end < now)
				estimated_end = now;
			estimated_transfer_length[i] = starpu_sched_component_transfer_length(c, task);
			estimated_ends_with_task[i] = compute_expected_time(now,
									    estimated_end,
									    estimated_lengths[i],
									    &estimated_transfer_length[i]);
			if(estimated_ends_with_task[i] < *min_exp_end_with_task)
				*min_exp_end_with_task = estimated_ends_with_task[i];
			if(estimated_ends_with_task[i] > *max_exp_end_with_task)
				*max_exp_end_with_task = estimated_ends_with_task[i];
			suitable_components[nsuitable_components++] = i;
		}
	}
	return nsuitable_components;
}
