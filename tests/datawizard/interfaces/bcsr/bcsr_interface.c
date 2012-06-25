/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
 * Copyright (C) 2011, 2012  inria
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
#include "../test_interfaces.h"
#include "../../../helper.h"

/*
 * In this test, we use the following matrix: 
 *
 *   +----------------+
 *   |  0   1   0   0 |
 *   |  2   3   0   0 |
 *   |  4   5   8   9 |
 *   |  6   7  10  11 |
 *   +----------------+
 *
 * nzval  = [0, 1, 2, 3] ++ [4, 5, 6, 7] ++ [8, 9, 10, 11]
 * colind = [0, 0, 1]
 * rowptr = [0, 1 ]
 * r = c = 2
 */

/* Size of the blocks */
#define R              2
#define C              2

#define NNZ_BLOCKS     3   /* out of 4 */
#define NZVAL_SIZE     (R*C*NNZ_BLOCKS)

#ifdef STARPU_USE_CPU
static void test_bcsr_cpu_func(void *buffers[], void *args);
#endif /* !STARPU_USE_CPU */
#ifdef STARPU_USE_CUDA
extern void test_bcsr_cuda_func(void *buffers[], void *_args);
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
extern void test_bcsr_opencl_func(void *buffers[], void *args);
#endif /* !STARPU_USE_OPENCL */


static int nzval[NZVAL_SIZE]  = {
	0, 1, 2, 3,    /* Fisrt block  */
	4, 5, 6, 7,    /* Second block */
	8, 9, 10, 11   /* Third block  */
};
static int nzval2[NZVAL_SIZE];

static uint32_t colind[NNZ_BLOCKS] = { 0, 0, 2 };
static uint32_t colind2[NNZ_BLOCKS];

static uint32_t rowptr[2] = { 0, 1 };
static uint32_t rowptr2[2];

static starpu_data_handle_t bcsr_handle;
static starpu_data_handle_t bcsr2_handle;


struct test_config bcsr_config =
{
#ifdef STARPU_USE_CPU
	.cpu_func      = test_bcsr_cpu_func,
#endif /* !STARPU_USE_CPU */
#ifdef STARPU_USE_CUDA
	.cuda_func     = test_bcsr_cuda_func,
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	.opencl_func   = test_bcsr_opencl_func,
#endif /* !STARPU_USE_OPENCL */
	.handle        = &bcsr_handle,
	.dummy_handle  = &bcsr2_handle,
	.copy_failed   = 0,
	.name          = "bcsr_interface"
};

static void
register_data(void)
{
	starpu_bcsr_data_register(&bcsr_handle,
				  0,
				  NNZ_BLOCKS,
				  1, /* nrow */
				  (uintptr_t) nzval,
				  colind,
				  rowptr,
				  0, /* firstentry */
				  R,
				  C,
				  sizeof(nzval[0]));

	starpu_bcsr_data_register(&bcsr2_handle,
				  0,
				  NNZ_BLOCKS,
				  1, /* nrow */
				  (uintptr_t) nzval2,
				  colind2,
				  rowptr2,
				  0, /* firstentry */
				  R,
				  C,
				  sizeof(nzval2[0]));
}

static void
unregister_data(void)
{
	starpu_data_unregister(bcsr_handle);
	starpu_data_unregister(bcsr2_handle);
}

static void
test_bcsr_cpu_func(void *buffers[], void *args)
{
	STARPU_SKIP_IF_VALGRIND;

	int *val;
	int factor;
	int i;

	uint32_t nnz = STARPU_BCSR_GET_NNZ(buffers[0]);
 	uint32_t r   = ((struct starpu_bcsr_interface *)buffers[0])->r;
 	uint32_t c   = ((struct starpu_bcsr_interface *)buffers[0])->c;
	if (r != R || c != C)
	{
		bcsr_config.copy_failed = 1;
		return;
	}
	nnz *= (r*c);

	val = (int *) STARPU_BCSR_GET_NZVAL(buffers[0]);
	factor = *(int *) args;

	for (i = 0; i < nnz; i++)
	{
		if (val[i] != i * factor)
		{
			bcsr_config.copy_failed = 1;
			return;
		}
		val[i] *= -1;
	}

#if 0
	/* TODO */
	/* Check colind */
	uint32_t *col = STARPU_BCSR_GET_COLIND(buffers[0]);
	for (i = 0; i < NNZ_BLOCKS; i++)
		if (col[i] != colind[i])
			bcsr_config.copy_failed = 1;

	/* Check rowptr */
	uint32_t *row = STARPU_BCSR_GET_ROWPTR(buffers[0]);
	for (i = 0; i < 1 + WIDTH/R; i++)
		if (row[i] != rowptr[i])
			bcsr_config.copy_failed = 1;
#endif
}

int
main(void)
{
	data_interface_test_summary *summary;
	struct starpu_conf conf;
	starpu_conf_init(&conf);

	conf.ncuda = 2;
	conf.nopencl = 1;

	if (starpu_init(&conf) == -ENODEV || starpu_cpu_worker_get_count() == 0)
		return STARPU_TEST_SKIPPED;

	register_data();

	summary = run_tests(&bcsr_config);
	if (!summary)
		exit(EXIT_FAILURE);

	unregister_data();

	starpu_shutdown();

	data_interface_test_summary_print(stderr, summary);

	return data_interface_test_summary_success(summary);
}
