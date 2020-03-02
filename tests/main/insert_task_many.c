/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_config.h>
#include "../helper.h"

/*
 * Try to pass many parameters to a task, testing the various codelet
 * declarations
 */

#define NPARAMS 15

void func_cpu(void *descr[], void *_args)
{
	(void)_args;
	struct starpu_task *task = starpu_task_get_current();
	int num = STARPU_TASK_GET_NBUFFERS(task);
	int i;

	for (i = 0; i < num; i++)
		if ((STARPU_TASK_GET_MODE(task, i) & STARPU_W)
		 || (STARPU_TASK_GET_MODE(task, i) & STARPU_SCRATCH))
		{
			int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[i]);

			*x = *x + 1;
		}
}

/* We will fill this one with dyn_modes */
struct starpu_codelet codelet_dyn =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = NPARAMS,
};

/* This will warn out at compilation time when maxbuffers is less than NPARAMS.
 * That is on purpose: we here check that we still behave correctly in that case.
 * We are just not able to check the parameter access modes.  */
struct starpu_codelet codelet_toomany =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = NPARAMS,
	.modes =
	{
		STARPU_R,
		STARPU_R,
		STARPU_RW|STARPU_COMMUTE,
		STARPU_RW|STARPU_COMMUTE,
		STARPU_R,
		STARPU_RW,
		STARPU_R,
		STARPU_RW|STARPU_COMMUTE,
		STARPU_R,
		STARPU_RW|STARPU_COMMUTE,
		STARPU_R,
		STARPU_R,
		STARPU_SCRATCH,
		STARPU_SCRATCH,
		STARPU_SCRATCH,
	}
};

struct starpu_codelet codelet_variable =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
};

int main(void)
{
        int *x;
        int i, ret, loop;

#ifdef STARPU_QUICK_CHECK
	int nloops = 4;
#else
	int nloops = 16;
#endif
	int val_int = 42;
	double val_double = 42.;
        starpu_data_handle_t *data_handles;
	int *expected;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	codelet_dyn.dyn_modes = malloc(NPARAMS * sizeof(*(codelet_dyn.modes)));
	codelet_dyn.dyn_modes[0] = STARPU_R,
	codelet_dyn.dyn_modes[1] = STARPU_R,
	codelet_dyn.dyn_modes[2] = STARPU_RW|STARPU_COMMUTE,
	codelet_dyn.dyn_modes[3] = STARPU_RW|STARPU_COMMUTE,
	codelet_dyn.dyn_modes[4] = STARPU_R,
	codelet_dyn.dyn_modes[5] = STARPU_RW,
	codelet_dyn.dyn_modes[6] = STARPU_R,
	codelet_dyn.dyn_modes[7] = STARPU_RW|STARPU_COMMUTE,
	codelet_dyn.dyn_modes[8] = STARPU_R,
	codelet_dyn.dyn_modes[9] = STARPU_RW|STARPU_COMMUTE,
	codelet_dyn.dyn_modes[10] = STARPU_R,
	codelet_dyn.dyn_modes[11] = STARPU_R,
	codelet_dyn.dyn_modes[12] = STARPU_SCRATCH,
	codelet_dyn.dyn_modes[13] = STARPU_SCRATCH,
	codelet_dyn.dyn_modes[14] = STARPU_SCRATCH,

	x = calloc(NPARAMS, sizeof(*x));
	data_handles = malloc(NPARAMS * sizeof(*data_handles));
	expected = calloc(NPARAMS, sizeof(*expected));
	for(i=0 ; i<NPARAMS ; i++)
		starpu_variable_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)&x[i], sizeof(x[i]));

	for (loop = 0; loop < nloops; loop++)
	{
		for (i = 0; i < NPARAMS; i++)
		{
			if (codelet_dyn.dyn_modes[i] & STARPU_W)
				expected[i]++;
		}
		ret = starpu_task_insert(&codelet_dyn,
					 STARPU_R, data_handles[0],
					 STARPU_R, data_handles[1],
					 STARPU_RW|STARPU_COMMUTE, data_handles[2],
					 STARPU_RW|STARPU_COMMUTE, data_handles[3],
					 STARPU_R, data_handles[4],
					 STARPU_RW, data_handles[5],
					 STARPU_R, data_handles[6],
					 STARPU_RW|STARPU_COMMUTE, data_handles[7],
					 STARPU_R, data_handles[8],
					 STARPU_RW|STARPU_COMMUTE, data_handles[9],
					 STARPU_R, data_handles[10],
					 STARPU_R, data_handles[11],
					 STARPU_SCRATCH, data_handles[12],
					 STARPU_SCRATCH, data_handles[13],
					 STARPU_SCRATCH, data_handles[14],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* Same, but using the toomany codelet */
		for (i = 0; i < NPARAMS; i++)
		{
			if (codelet_dyn.dyn_modes[i] & STARPU_W)
				expected[i]++;
		}
		ret = starpu_task_insert(&codelet_toomany,
					 STARPU_R, data_handles[0],
					 STARPU_R, data_handles[1],
					 STARPU_RW|STARPU_COMMUTE, data_handles[2],
					 STARPU_RW|STARPU_COMMUTE, data_handles[3],
					 STARPU_R, data_handles[4],
					 STARPU_RW, data_handles[5],
					 STARPU_R, data_handles[6],
					 STARPU_RW|STARPU_COMMUTE, data_handles[7],
					 STARPU_R, data_handles[8],
					 STARPU_RW|STARPU_COMMUTE, data_handles[9],
					 STARPU_R, data_handles[10],
					 STARPU_R, data_handles[11],
					 STARPU_SCRATCH, data_handles[12],
					 STARPU_SCRATCH, data_handles[13],
					 STARPU_SCRATCH, data_handles[14],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* Same, but using the variable codelet */
		for (i = 0; i < NPARAMS; i++)
		{
			if (codelet_dyn.dyn_modes[i] & STARPU_W)
				expected[i]++;
		}
		ret = starpu_task_insert(&codelet_variable,
					 STARPU_R, data_handles[0],
					 STARPU_R, data_handles[1],
					 STARPU_RW|STARPU_COMMUTE, data_handles[2],
					 STARPU_RW|STARPU_COMMUTE, data_handles[3],
					 STARPU_R, data_handles[4],
					 STARPU_RW, data_handles[5],
					 STARPU_R, data_handles[6],
					 STARPU_RW|STARPU_COMMUTE, data_handles[7],
					 STARPU_R, data_handles[8],
					 STARPU_RW|STARPU_COMMUTE, data_handles[9],
					 STARPU_R, data_handles[10],
					 STARPU_R, data_handles[11],
					 STARPU_SCRATCH, data_handles[12],
					 STARPU_SCRATCH, data_handles[13],
					 STARPU_SCRATCH, data_handles[14],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	}

enodev:
        for(i=0 ; i<NPARAMS ; i++)
	{
                starpu_data_unregister(data_handles[i]);
        }

	starpu_shutdown();
	free(data_handles);
	free(codelet_dyn.dyn_modes);

	if (ret == -ENODEV)
	{
		fprintf(stderr, "WARNING: No one can execute this task\n");
		/* yes, we do not perform the computation but we did detect that no one
		 * could perform the kernel, so this is not an error from StarPU */
		free(x);
		free(expected);
		return STARPU_TEST_SKIPPED;
	}
	else
	{
		for(i=0 ; i<NPARAMS; i++)
		{
			if (x[i] != expected[i])
			{
				FPRINTF(stderr, "[end loop] value[%d] = %d != Expected value %d\n", i, x[i], expected[i]);
				ret = 1;
			}
		}
		if (ret == 0)
		{
			FPRINTF(stderr, "[end of loop] all values are correct\n");
		}
		free(x);
		free(expected);
		return ret;
	}
}
