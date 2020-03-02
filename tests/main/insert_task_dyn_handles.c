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
 * Try the starpu_task_insert interface in various ways, and notably
 * triggering the use of dyn_handles
 */

void func_cpu(void *descr[], void *_args)
{
	int num = STARPU_TASK_GET_NBUFFERS(starpu_task_get_current());
	int i;

	(void)_args;

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

struct starpu_codelet codelet_minus1 =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = STARPU_NMAXBUFS-1,
};

struct starpu_codelet codelet_exactly =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = STARPU_NMAXBUFS,
};

struct starpu_codelet codelet_plus1 =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = STARPU_NMAXBUFS+1,
};

struct starpu_codelet codelet_plus5 =
{
	.cpu_funcs = {func_cpu},
	/* starpu_task_get_current() doesn't work on MIC */
	/* .cpu_funcs_name = {"func_cpu"}, */
	.nbuffers = STARPU_NMAXBUFS+5,
};

starpu_data_handle_t *data_handles;
struct starpu_data_descr *descrs;
int *expected;

int test(int n, struct starpu_codelet *static_codelet)
{
	int i, ret;

	for (i = 0; i < n; i++)
		expected[i]++;
	ret = starpu_task_insert(&codelet,
				 STARPU_DATA_MODE_ARRAY, descrs, n,
				 0);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Same with static number of buffers in codelet */
	for (i = 0; i < n; i++)
		expected[i]++;
	ret = starpu_task_insert(static_codelet,
				 STARPU_DATA_MODE_ARRAY, descrs, n,
				 0);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* Test a whole array after one data */
	expected[0]++;
	for (i = 1; i < n; i++)
		expected[i]++;
	ret = starpu_task_insert(&codelet,
				 STARPU_RW, data_handles[0],
				 STARPU_DATA_MODE_ARRAY, &descrs[1], n-1,
				 0);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	if (n > 1)
	{
		/* Same with static number of buffers in codelet */
		expected[0]++;
		for (i = 1; i < n; i++)
			expected[i]++;
		ret = starpu_task_insert(static_codelet,
					 STARPU_RW, data_handles[0],
					 STARPU_DATA_MODE_ARRAY, &descrs[1], n-1,
					 0);
		if (ret == -ENODEV) return ret;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	return 0;
}

int main(void)
{
        int *x;
        int i, ret, loop;

#ifdef STARPU_QUICK_CHECK
	int nloops = 4;
#else
	int nloops = 16;
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	x = calloc(STARPU_NMAXBUFS+5, sizeof(*x));
	data_handles = malloc((STARPU_NMAXBUFS+5) * sizeof(*data_handles));
	descrs = malloc((STARPU_NMAXBUFS+5) * sizeof(*descrs));
	expected = calloc(STARPU_NMAXBUFS+5, sizeof(*expected));
	for(i=0 ; i<STARPU_NMAXBUFS+5 ; i++)
	{
		starpu_variable_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)&x[i], sizeof(x[i]));
		descrs[i].handle = data_handles[i];
		descrs[i].mode = STARPU_RW;
	}

	for (loop = 0; loop < nloops; loop++)
	{
		/* Test smaller than NMAXBUFS */
		ret = test(STARPU_NMAXBUFS-1, &codelet_minus1);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* Test exactly NMAXBUFS */
		ret = test(STARPU_NMAXBUFS, &codelet_exactly);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* Test more than NMAXBUFS */
		ret = test(STARPU_NMAXBUFS+1, &codelet_plus1);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* Test yet more than NMAXBUFS */
		ret = test(STARPU_NMAXBUFS+5, &codelet_plus5);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* Test datas one after the other, but less than NMAXBUFS */
		expected[0]++;
		for (i = 1; i < STARPU_NMAXBUFS-1 && i < 10; i++)
			expected[i]++;
		ret = starpu_task_insert(&codelet,
					 STARPU_RW, data_handles[0],
#if STARPU_NMAXBUFS > 2
					 STARPU_RW, data_handles[1],
#endif
#if STARPU_NMAXBUFS > 3
					 STARPU_RW, data_handles[2],
#endif
#if STARPU_NMAXBUFS > 4
					 STARPU_RW, data_handles[3],
#endif
#if STARPU_NMAXBUFS > 5
					 STARPU_RW, data_handles[4],
#endif
#if STARPU_NMAXBUFS > 6
					 STARPU_RW, data_handles[5],
#endif
#if STARPU_NMAXBUFS > 7
					 STARPU_RW, data_handles[6],
#endif
#if STARPU_NMAXBUFS > 8
					 STARPU_RW, data_handles[7],
#endif
#if STARPU_NMAXBUFS > 9
					 STARPU_RW, data_handles[8],
#endif
#if STARPU_NMAXBUFS > 10
					 STARPU_RW, data_handles[9],
#endif
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

#if STARPU_NMAXBUFS > 1 && STARPU_NMAXBUFS <= 8
		/* Same with static number of buffers in codelet */
		expected[0]++;
		for (i = 1; i < STARPU_NMAXBUFS-1 && i < 7; i++)
			expected[i]++;
		ret = starpu_task_insert(&codelet_minus1,
					 STARPU_RW, data_handles[0],
#if STARPU_NMAXBUFS > 2
					 STARPU_RW, data_handles[1],
#endif
#if STARPU_NMAXBUFS > 3
					 STARPU_RW, data_handles[2],
#endif
#if STARPU_NMAXBUFS > 4
					 STARPU_RW, data_handles[3],
#endif
#if STARPU_NMAXBUFS > 5
					 STARPU_RW, data_handles[4],
#endif
#if STARPU_NMAXBUFS > 6
					 STARPU_RW, data_handles[5],
#endif
#if STARPU_NMAXBUFS > 7
					 STARPU_RW, data_handles[6],
#endif
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
#endif




		/* Test datas one after the other, but more than NMAXBUFS */
		for (i = 0; i < STARPU_NMAXBUFS+5 && i < 10; i++)
			expected[i]++;
		ret = starpu_task_insert(&codelet,
					 STARPU_RW, data_handles[0],
					 STARPU_RW, data_handles[1],
					 STARPU_RW, data_handles[2],
					 STARPU_RW, data_handles[3],
					 STARPU_RW, data_handles[4],
					 STARPU_RW, data_handles[5],
#if STARPU_NMAXBUFS > 1
					 STARPU_RW, data_handles[6],
#endif
#if STARPU_NMAXBUFS > 2
					 STARPU_RW, data_handles[7],
#endif
#if STARPU_NMAXBUFS > 3
					 STARPU_RW, data_handles[8],
#endif
#if STARPU_NMAXBUFS > 4
					 STARPU_RW, data_handles[9],
#endif
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

#if STARPU_NMAXBUFS > 1 && STARPU_NMAXBUFS <= 8
		/* Same with static number of buffers in codelet*/
		for (i = 0; i < STARPU_NMAXBUFS+5 && i < 13; i++)
			expected[i]++;
		ret = starpu_task_insert(&codelet_plus5,
					 STARPU_RW, data_handles[0],
					 STARPU_RW, data_handles[1],
					 STARPU_RW, data_handles[2],
					 STARPU_RW, data_handles[3],
					 STARPU_RW, data_handles[4],
					 STARPU_RW, data_handles[5],
#if STARPU_NMAXBUFS > 1
					 STARPU_RW, data_handles[6],
#endif
#if STARPU_NMAXBUFS > 2
					 STARPU_RW, data_handles[7],
#endif
#if STARPU_NMAXBUFS > 3
					 STARPU_RW, data_handles[8],
#endif
#if STARPU_NMAXBUFS > 4
					 STARPU_RW, data_handles[9],
#endif
#if STARPU_NMAXBUFS > 5
					 STARPU_RW, data_handles[10],
#endif
#if STARPU_NMAXBUFS > 6
					 STARPU_RW, data_handles[11],
#endif
#if STARPU_NMAXBUFS > 7
					 STARPU_RW, data_handles[12],
#endif
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
#endif

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
		free(expected);
		return STARPU_TEST_SKIPPED;
	}
	else
	{
		for(i=0 ; i<STARPU_NMAXBUFS+5; i++)
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
