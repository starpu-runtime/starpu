/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	{
		if (i % 4 == 3)
		{
			assert(STARPU_TASK_GET_NODE(task, i, -42) == STARPU_SPECIFIC_NODE_CPU);
		}
		else
		{
			assert(STARPU_TASK_GET_NODE(task, i, -42) == STARPU_SPECIFIC_NODE_LOCAL);
		}
		if ((STARPU_TASK_GET_MODE(task, i) & STARPU_W)
		 || (STARPU_TASK_GET_MODE(task, i) & STARPU_SCRATCH))
		{
			int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[i]);

			*x = *x + 1;
		}
	}
}

/* We will fill this one with dyn_modes */
struct starpu_codelet codelet_dyn =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MPI server client mode */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = NPARAMS,
	.specific_nodes = 1
};

struct starpu_codelet codelet_variable =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MPI server client mode */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
};

int main(void)
{
	int *x;
	int i, ret, loop;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

#ifdef STARPU_QUICK_CHECK
	int nloops = 4;
#else
	int nloops = 16;
#endif
	int val_int = 42;
	double val_double = 42.;
	starpu_data_handle_t *data_handles;
	int *expected;

	ret = starpu_init(&conf);
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

	codelet_dyn.dyn_nodes = malloc(NPARAMS * sizeof(*(codelet_dyn.modes)));
	codelet_dyn.dyn_nodes[0] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[1] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[2] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[3] =  STARPU_SPECIFIC_NODE_CPU,
	codelet_dyn.dyn_nodes[4] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[5] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[6] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[7] =  STARPU_SPECIFIC_NODE_CPU,
	codelet_dyn.dyn_nodes[8] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[9] =  STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[10] = STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[11] = STARPU_SPECIFIC_NODE_CPU,
	codelet_dyn.dyn_nodes[12] = STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[13] = STARPU_SPECIFIC_NODE_LOCAL,
	codelet_dyn.dyn_nodes[14] = STARPU_SPECIFIC_NODE_LOCAL,

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
		struct starpu_task *task = starpu_task_build(&codelet_variable,
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
		task->specific_nodes = 1;
		task->dyn_nodes = malloc(NPARAMS * sizeof(*(codelet_dyn.modes)));
	        task->dyn_nodes[0] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[1] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[2] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[3] =  STARPU_SPECIFIC_NODE_CPU;
	        task->dyn_nodes[4] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[5] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[6] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[7] =  STARPU_SPECIFIC_NODE_CPU;
	        task->dyn_nodes[8] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[9] =  STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[10] = STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[11] = STARPU_SPECIFIC_NODE_CPU;
	        task->dyn_nodes[12] = STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[13] = STARPU_SPECIFIC_NODE_LOCAL;
	        task->dyn_nodes[14] = STARPU_SPECIFIC_NODE_LOCAL;
		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* Same, but using the toomany codelet directly with starpu_task_insert */
		for (i = 0; i < NPARAMS; i++)
		{
			if (codelet_dyn.dyn_modes[i] & STARPU_W)
				expected[i]++;
		}
		struct starpu_data_mode_node_descr descrs[NPARAMS];
		descrs[0].handle = data_handles[0];
		descrs[0].mode = STARPU_R;
		descrs[0].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[1].handle = data_handles[1];
		descrs[1].mode = STARPU_R;
		descrs[1].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[2].handle = data_handles[2];
		descrs[2].mode = STARPU_RW|STARPU_COMMUTE;
		descrs[2].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[3].handle = data_handles[3];
		descrs[3].mode = STARPU_RW|STARPU_COMMUTE;
		descrs[3].node = STARPU_SPECIFIC_NODE_CPU;
		descrs[4].handle = data_handles[4];
		descrs[4].mode = STARPU_R;
		descrs[4].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[5].handle = data_handles[5];
		descrs[5].mode = STARPU_RW;
		descrs[5].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[6].handle = data_handles[6];
		descrs[6].mode = STARPU_R;
		descrs[6].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[7].handle = data_handles[7];
		descrs[7].mode = STARPU_RW|STARPU_COMMUTE;
		descrs[7].node = STARPU_SPECIFIC_NODE_CPU;
		descrs[8].handle = data_handles[8];
		descrs[8].mode = STARPU_R;
		descrs[8].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[9].handle = data_handles[9];
		descrs[9].mode = STARPU_RW|STARPU_COMMUTE;
		descrs[9].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[10].handle = data_handles[10];
		descrs[10].mode = STARPU_R;
		descrs[10].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[11].handle = data_handles[11];
		descrs[11].mode = STARPU_R;
		descrs[11].node = STARPU_SPECIFIC_NODE_CPU;
		descrs[12].handle = data_handles[12];
		descrs[12].mode = STARPU_SCRATCH;
		descrs[12].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[13].handle = data_handles[13];
		descrs[13].mode = STARPU_SCRATCH;
		descrs[13].node = STARPU_SPECIFIC_NODE_LOCAL;
		descrs[14].handle = data_handles[14];
		descrs[14].mode = STARPU_SCRATCH;
		descrs[14].node = STARPU_SPECIFIC_NODE_LOCAL;

		ret = starpu_task_insert(&codelet_variable,
					 STARPU_DATA_MODE_NODE_ARRAY, descrs, NPARAMS,
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
