/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013, 2014  CNRS
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
#include <starpu_config.h>
#include "../helper.h"

void func_cpu(void *descr[], void *_args)
{
	int num = starpu_task_get_current()->nbuffers;
	int i;

	for (i = 0; i < num; i++)
	{
		int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[i]);

		*x = *x + 1;
	}
}

struct starpu_codelet codelet =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
};

int main(int argc, char **argv)
{
        int *x;
        int i, ret, loop;

#ifdef STARPU_QUICK_CHECK
	int nloops = 4;
#else
	int nloops = 16;
#endif
        starpu_data_handle_t *data_handles;
	struct starpu_data_descr *descrs;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	x = calloc(1, (STARPU_NMAXBUFS+5) * sizeof(int));
	data_handles = malloc((STARPU_NMAXBUFS+5) * sizeof(starpu_data_handle_t));
	descrs = malloc((STARPU_NMAXBUFS+5) * sizeof(struct starpu_data_descr));
	for(i=0 ; i<STARPU_NMAXBUFS+5 ; i++)
	{
		starpu_variable_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)&x[i], sizeof(x[i]));
		descrs[i].handle = data_handles[i];
		descrs[i].mode = STARPU_RW;
	}

	for (loop = 0; loop < nloops; loop++)
	{
		ret = starpu_task_insert(&codelet,
					 STARPU_DATA_MODE_ARRAY, descrs, STARPU_NMAXBUFS-1,
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		ret = starpu_task_insert(&codelet,
					 STARPU_DATA_MODE_ARRAY, descrs, STARPU_NMAXBUFS+5,
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

enodev:
        for(i=0 ; i<STARPU_NMAXBUFS+5 ; i++)
	{
                starpu_data_unregister(data_handles[i]);
        }

	starpu_shutdown();
	free(data_handles);
	free(descrs);

	if (ret == -ENODEV)
	{
		fprintf(stderr, "WARNING: No one can execute this task\n");
		/* yes, we do not perform the computation but we did detect that no one
		 * could perform the kernel, so this is not an error from StarPU */
		free(x);
		return STARPU_TEST_SKIPPED;
	}
	else
	{
		for(i=0 ; i<STARPU_NMAXBUFS-1 ; i++)
		{
			if (x[i] != nloops * 2)
			{
				FPRINTF(stderr, "[end loop] value[%d] = %d != Expected value %d\n", i, x[i], nloops*2);
				ret = 1;
			}
		}
		for(i=STARPU_NMAXBUFS-1 ; i<STARPU_NMAXBUFS+5 ; i++)
		{
			if (x[i] != nloops)
			{
				FPRINTF(stderr, "[end loop] value[%d] = %d != Expected value %d\n", i, x[i], nloops);
				ret = 1;
			}
		}
		if (ret == 0)
		{
			FPRINTF(stderr, "[end of loop] all values are correct\n");
		}
		free(x);
		return ret;
	}
}
