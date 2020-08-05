/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

static starpu_data_handle_t bcsr_handle;

void cpu_show_bcsr(void *descr[], void *arg)
{
	(void)arg;
	struct starpu_bcsr_interface *iface = descr[0];
	uint32_t nnz = iface->nnz;
	uint32_t nrow = iface->nrow;
	int *nzval = (int *)iface->nzval;
	uint32_t *colind = iface->colind;
	uint32_t *rowptr = iface->rowptr;

	uint32_t firstentry = iface->firstentry;
	uint32_t r = iface->r;
	uint32_t c = iface->c;
	uint32_t elemsize = iface->elemsize;

	uint32_t i, j, y, x;
	static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);

	printf("\nnnz %u elemsize %u\n", nnz, elemsize);

	for (i = 0; i < nrow; i++)
	{
		uint32_t row_start = rowptr[i] - firstentry;
		uint32_t row_end = rowptr[i+1] - firstentry;

		printf("row %u\n", i);

		for (j = row_start; j < row_end; j++)
		{
			int *block = nzval + j * r*c;

			printf( " column %u\n", colind[j]);

			for (y = 0; y < r; y++)
			{
				for (x = 0; x < c; x++)
					printf("  %d", block[y*c+x]);
				printf("\n");
			}
		}
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}


struct starpu_codelet show_cl =
{
	.cpu_funcs = { cpu_show_bcsr },
	.nbuffers = 1,
	.modes = { STARPU_R },
};

/*
 * In this test, we use the following matrix:
 *
 *   +----------------+
 *   |  0   1   0   0 |
 *   |  2   3   0   0 |
 *   |  4   5   8   9 |
 *   |  6   7  10  11 |
 *   |  0   0   0   0 |
 *   |  0   0   0   0 |
 *   +----------------+
 *
 * nzval  = [0, 1, 2, 3] ++ [4, 5, 6, 7] ++ [8, 9, 10, 11]
 * colind = [0, 0, 1] (column index of each non-zero block)
 * rowptr = [0, 1, 3] (index of first non-zero block for each row)
 * r = c = 2
 */

/* Size of the blocks */
#define R              2
#define C              2

#define NNZ_BLOCKS     3   /* out of 6 */
#define NZVAL_SIZE     (R*C*NNZ_BLOCKS)

#define NROWS          3

static int nzval[NZVAL_SIZE]  =
{
	0, 1, 2, 3,    /* First block  */
	4, 5, 6, 7,    /* Second block */
	8, 9, 10, 11   /* Third block  */
};
static uint32_t colind[NNZ_BLOCKS] = { 0, 0, 1 };

static uint32_t rowptr[NROWS+1] = { 0, 1, NNZ_BLOCKS, NNZ_BLOCKS };

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	starpu_conf_init(&conf);

	conf.precedence_over_environment_variables = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;

	if (starpu_initialize(&conf, &argc, &argv) == -ENODEV)
		return STARPU_TEST_SKIPPED;

	if (starpu_cpu_worker_get_count() == 0 || starpu_memory_nodes_get_count() > 1) {
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	starpu_bcsr_data_register(&bcsr_handle,
				  STARPU_MAIN_RAM,
				  NNZ_BLOCKS,
				  NROWS,
				  (uintptr_t) nzval,
				  colind,
				  rowptr,
				  0, /* firstentry */
				  R,
				  C,
				  sizeof(nzval[0]));

	starpu_task_insert(&show_cl, STARPU_R, bcsr_handle, 0);

	struct starpu_data_filter filter =
	{
		.filter_func = starpu_bcsr_filter_vertical_block,
		.nchildren = 3,
	};
	starpu_data_partition(bcsr_handle, &filter);

	starpu_task_insert(&show_cl, STARPU_R, starpu_data_get_sub_data(bcsr_handle, 1, 0), 0);
	starpu_task_insert(&show_cl, STARPU_R, starpu_data_get_sub_data(bcsr_handle, 1, 1), 0);
	starpu_task_insert(&show_cl, STARPU_R, starpu_data_get_sub_data(bcsr_handle, 1, 2), 0);

	starpu_data_unpartition(bcsr_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(bcsr_handle);

	starpu_shutdown();

	return 0;
}
