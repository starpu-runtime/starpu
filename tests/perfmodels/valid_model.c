/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
#include <core/perfmodel/perfmodel.h>
#include <unistd.h>
#include "../helper.h"

/*
 * Check that measurements get recorded in the performance model
 */

void func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	usleep(1000);
}

static struct starpu_perfmodel rb_model =
{
	.type = STARPU_REGRESSION_BASED,
	.symbol = "valid_model_regression_based"
};

static struct starpu_perfmodel nlrb_model =
{
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "valid_model_non_linear_regression_based"
};

#if 0
static struct starpu_perfmodel hb_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "valid_model_history_based"
};
#endif

static struct starpu_codelet mycodelet =
{
	.cuda_funcs = {func},
	.opencl_funcs = {func},
	.cpu_funcs = {func},
	.cpu_funcs_name = {"func"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static int submit(struct starpu_codelet *codelet, struct starpu_perfmodel *model)
{
	int nloops = 123;
	int loop;
	starpu_data_handle_t handle;
	struct starpu_perfmodel lmodel;
	int ret;
	int old_nsamples, new_nsamples;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.sched_policy_name = "eager";
	conf.calibrate = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	codelet->model = model;

	old_nsamples = 0;
	memset(&lmodel, 0, sizeof(struct starpu_perfmodel));
	lmodel.type = model->type;
	ret = starpu_perfmodel_load_symbol(codelet->model->symbol, &lmodel);
	if (ret != 1)
	{
		int i, impl;
		for(i = 0; i < lmodel.state->ncombs; i++)
		{
			int comb = lmodel.state->combs[i];
			for(impl = 0; impl < lmodel.state->nimpls[comb]; impl++)
				old_nsamples += lmodel.state->per_arch[comb][impl].regression.nsample;
		}
	}

        starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, 100, sizeof(int));
	for (loop = 0; loop < nloops; loop++)
	{
		ret = starpu_task_insert(codelet, STARPU_W, handle, 0);
		if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
        starpu_data_unregister(handle);
	starpu_perfmodel_unload_model(&lmodel);
	starpu_shutdown(); // To force dumping perf models on disk

	// We need to call starpu_init again to initialise values used by perfmodels
	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	char path[256];
	starpu_perfmodel_get_model_path(codelet->model->symbol, path, 256);
	FPRINTF(stderr, "Perfmodel File <%s>\n", path);
	ret = starpu_perfmodel_load_file(path, &lmodel);

	if (ret == 1)
	{
		FPRINTF(stderr, "The performance model for the symbol <%s> could not be loaded\n", codelet->model->symbol);
		starpu_shutdown();
		return 1;
	}
	else
	{
		int i;
		new_nsamples = 0;
		for(i = 0; i < lmodel.state->ncombs; i++)
		{
			int comb = lmodel.state->combs[i];
			int impl;
			for(impl = 0; impl < lmodel.state->nimpls[comb]; impl++)
			     new_nsamples += lmodel.state->per_arch[comb][impl].regression.nsample;
		}
	}

	ret = starpu_perfmodel_unload_model(&lmodel);
	starpu_shutdown();
	if (ret == 1)
	{
		FPRINTF(stderr, "The performance model for the symbol <%s> could not be UNloaded\n", codelet->model->symbol);
		return 1;
	}

	if (old_nsamples + nloops == new_nsamples)
	{
		FPRINTF(stderr, "Sampling for <%s> OK %d + %d == %d\n", codelet->model->symbol, old_nsamples, nloops, new_nsamples);
		return EXIT_SUCCESS;
	}
	else
	{
		FPRINTF(stderr, "Sampling for <%s> failed %d + %d != %d\n", codelet->model->symbol, old_nsamples, nloops, new_nsamples);
		return EXIT_FAILURE;
	}
}

int main(void)
{
	int ret;

	/* Use a linear regression model */
	ret = submit(&mycodelet, &rb_model);
	if (ret) return ret;

	/* Use a non-linear regression model */
	ret = submit(&mycodelet, &nlrb_model);
	if (ret) return ret;

#ifdef STARPU_DEVEL
#  warning history based model cannot be validated with regression.nsample
#endif
#if 0
	/* Use a history model */
	ret = submit(&mycodelet, &hb_model);
	if (ret) return ret;
#endif

	return EXIT_SUCCESS;
}
