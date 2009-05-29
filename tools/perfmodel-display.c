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

#include <starpu.h>
#include <assert.h>
#include <stdio.h>

#include <starpu-perfmodel.h>

static struct starpu_perfmodel_t model;

static void display_perf_model(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch)
{
	struct starpu_per_arch_perfmodel_t *arch_model = &model->per_arch[arch];

	fprintf(stderr, "\tRegression : #sample = %d (%s)\n",
		arch_model->regression.nsample, arch_model->regression.valid?"VALID":"INVALID");

	/* Only display the regression model if we could actually build a model */
	if (arch_model->regression.valid)
	{
		fprintf(stderr, "\tLinear: y = alpha size ^ beta\n");
		fprintf(stderr, "\t\talpha = %le\n", arch_model->regression.alpha);
		fprintf(stderr, "\t\tbeta = %le\n", arch_model->regression.beta);

		fprintf(stderr, "\tNon-Linear: y = a size ^b + c\n");
		fprintf(stderr, "\t\ta = %le\n", arch_model->regression.a);
		fprintf(stderr, "\t\tb = %le\n", arch_model->regression.b);
		fprintf(stderr, "\t\tc = %le\n", arch_model->regression.c);
	}
}

static void display_all_perf_models(struct starpu_perfmodel_t *model)
{
	/* yet, we assume there is a single performance model per architecture */
	fprintf(stderr, "performance model for CPUs :\n");
	display_perf_model(model, STARPU_CORE_DEFAULT);

	fprintf(stderr, "performance model for CUDA :\n");
	display_perf_model(model, STARPU_CUDA_DEFAULT);

	fprintf(stderr, "performance model for GORDON :\n");
	display_perf_model(model, STARPU_GORDON_DEFAULT);
}

int main(int argc, char **argv)
{
	assert(argc == 2);

	const char *symbol = argv[1];

	fprintf(stderr, "symbol : %s\n", symbol);

	int ret = starpu_load_history_debug(symbol, &model);
	if (ret == 1)
	{
		fprintf(stderr, "The performance model could not be loaded\n");
		return 1;
	}

	fprintf(stderr, "Performance loaded\n");

	display_all_perf_models(&model);

	return 0;
}
