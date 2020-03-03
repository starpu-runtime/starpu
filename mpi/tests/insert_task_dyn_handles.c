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
#include <starpu_mpi.h>
#include <starpu_config.h>
#include "helper.h"

#define FFACTOR 42

void func_cpu(void *descr[], void *_args)
{
	int num = starpu_task_get_current()->nbuffers;
	int *factor = (int *)STARPU_VARIABLE_GET_PTR(descr[num-1]);
	int i;

	for (i = 0; i < num-1; i++)
	{
		int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[i]);

		*x = *x + 1**factor;
	}
}

#ifdef STARPU_SIMGRID
/* Dummy cost function for simgrid */
static double cost_function(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, unsigned nimpl STARPU_ATTRIBUTE_UNUSED)
{
	return 0.000001;
}
static struct starpu_perfmodel dumb_model =
{
	.type		= STARPU_COMMON,
	.cost_function	= cost_function
};
#endif

struct starpu_codelet codelet =
{
	.cpu_funcs = {func_cpu},
	.cpu_funcs_name = {"func_cpu"},
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
#ifdef STARPU_SIMGRID
	.model = &dumb_model,
#endif
};

int main(int argc, char **argv)
{
        int *x;
        int i, ret, loop;
	int rank;
	int factor=0;

#ifdef STARPU_QUICK_CHECK
	int nloops = 4;
#else
	int nloops = 16;
#endif
        starpu_data_handle_t *data_handles;
        starpu_data_handle_t factor_handle;
	struct starpu_data_descr *descrs;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	x = calloc(1, (STARPU_NMAXBUFS+15) * sizeof(int));
	data_handles = malloc((STARPU_NMAXBUFS+15) * sizeof(starpu_data_handle_t));
	descrs = malloc((STARPU_NMAXBUFS+15) * sizeof(struct starpu_data_descr));
	for(i=0 ; i<STARPU_NMAXBUFS+15 ; i++)
	{
		starpu_variable_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)&x[i], sizeof(x[i]));
		starpu_mpi_data_register(data_handles[i], i, 0);
		descrs[i].handle = data_handles[i];
		descrs[i].mode = STARPU_RW;
	}
	if (rank == 1)
		factor=FFACTOR;
	starpu_variable_data_register(&factor_handle, STARPU_MAIN_RAM, (uintptr_t)&factor, sizeof(factor));
	starpu_mpi_data_register(factor_handle, FFACTOR, 1);

	for (loop = 0; loop < nloops; loop++)
	{
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet,
					     STARPU_DATA_MODE_ARRAY, descrs, STARPU_NMAXBUFS-1,
					     STARPU_R, factor_handle,
					     0);
		if (ret == -ENODEV)
			goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");

		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet,
					     STARPU_DATA_MODE_ARRAY, descrs, STARPU_NMAXBUFS+15,
					     STARPU_R, factor_handle,
					     0);
		if (ret == -ENODEV)
			goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	}

enodev:
        for(i=0 ; i<STARPU_NMAXBUFS+15 ; i++)
	{
                starpu_data_unregister(data_handles[i]);
        }
	starpu_data_unregister(factor_handle);

	free(data_handles);
	free(descrs);

	if (ret == -ENODEV)
	{
		fprintf(stderr, "WARNING: No one can execute this task\n");
		/* yes, we do not perform the computation but we did detect that no one
		 * could perform the kernel, so this is not an error from StarPU */
		free(x);
		ret = STARPU_TEST_SKIPPED;
	}
	else if (rank == 0)
	{
		for(i=0 ; i<STARPU_NMAXBUFS-1 ; i++)
		{
			if (x[i] != nloops * FFACTOR * 2)
			{
				FPRINTF_MPI(stderr, "[end loop] value[%d] = %d != Expected value %d\n", i, x[i], nloops*2);
				ret = 1;
			}
		}
		for(i=STARPU_NMAXBUFS-1 ; i<STARPU_NMAXBUFS+15 ; i++)
		{
			if (x[i] != nloops * FFACTOR)
			{
				FPRINTF_MPI(stderr, "[end loop] value[%d] = %d != Expected value %d\n", i, x[i], nloops);
				ret = 1;
			}
		}
		if (ret == 0)
		{
			FPRINTF_MPI(stderr, "[end of loop] all values are correct\n");
		}
		free(x);
	}
	else
	{
		FPRINTF_MPI(stderr, "[end of loop] no computation on this node\n");
		ret = 0;
		free(x);
	}

	starpu_mpi_shutdown();
	starpu_shutdown();
	MPI_Finalize();
	return ret;
}
