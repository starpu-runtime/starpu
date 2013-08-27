/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012, 2013  Centre National de la Recherche Scientifique
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

#include <config.h>
#include <starpu.h>
#include "../helper.h"

static void func(void *descr[], void *arg)
{
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
	.cuda_funcs = {func, NULL},
	.opencl_funcs = {func, NULL},
	.cpu_funcs = {func, NULL},
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
	unsigned archid, archtype, devid, ncore;

	starpu_conf_init(&conf);
	conf.sched_policy_name = "eager";
	conf.calibrate = 1;

	initialize_model(model);

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	codelet->model = model;

	old_nsamples = 0;
	ret = starpu_perfmodel_load_symbol(codelet->model->symbol, &lmodel);
	if (ret != 1)
		for (archtype = 0; archtype < STARPU_NARCH; archtype++)
			if(lmodel.per_arch[archtype] != NULL)
				for(devid=0; lmodel.per_arch[archtype][devid] != NULL; devid++)
					for(ncore=0; lmodel.per_arch[archtype][devid][ncore] != NULL; ncore++)
						old_nsamples += lmodel.per_arch[archtype][devid][ncore][0].regression.nsample;

        starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, 100, sizeof(int));
	for (loop = 0; loop < nloops; loop++)
	{
		ret = starpu_insert_task(codelet, STARPU_W, handle, 0);
		if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
        starpu_data_unregister(handle);
	starpu_perfmodel_unload_model(&lmodel);
	starpu_shutdown(); // To force dumping perf models on disk

	ret = starpu_perfmodel_load_symbol(codelet->model->symbol, &lmodel);
	if (ret == 1)
	{
		FPRINTF(stderr, "The performance model for the symbol <%s> could not be loaded\n", codelet->model->symbol);
		return 1;
	}

	new_nsamples = 0;
	for (archtype = 0; archtype < STARPU_NARCH; archtype++)
		if(lmodel.per_arch[archtype] != NULL)
			for(devid=0; lmodel.per_arch[archtype][devid] != NULL; devid++)
				for(ncore=0; lmodel.per_arch[archtype][devid][ncore] != NULL; ncore++)
					new_nsamples += lmodel.per_arch[archtype][devid][ncore][0].regression.nsample;

	ret = starpu_perfmodel_unload_model(&lmodel);
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

int main(int argc, char **argv)
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
