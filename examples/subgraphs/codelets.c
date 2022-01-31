/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void matrix_fill(void *buffers[], void *cl_arg)
{
	unsigned i, j;
	(void)cl_arg;

	/* length of the matrix */
	unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	for(j=0; j<ny ; j++)
	{
		for(i=0; i<nx ; i++)
			val[(j*ld)+i] = i+100*j;
	}
}

struct starpu_codelet cl_fill =
{
	.cpu_funcs = {matrix_fill},
	.cpu_funcs_name = {"matrix_fill"},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.name = "matrix_fill"
};

void fmultiple_check_scale(void *buffers[], void *cl_arg)
{
	int start, factor;
	unsigned i, j;

	/* length of the matrix */
	unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	starpu_codelet_unpack_args(cl_arg, &start, &factor);

	char message[1024];
	int written=0;
	for(j=0; j<ny ; j++)
	{
		written += snprintf(message+written, sizeof(message)-written, "[%d]", j);
		for(i=0; i<nx ; i++)
		{
			written += snprintf(message+written, sizeof(message)-written, "%4d", val[(j*ld)+i]);
			STARPU_ASSERT(val[(j*ld)+i] == start + factor*((int)(i+100*j)));
			val[(j*ld)+i] *= 2;
		}
		written += snprintf(message+written, sizeof(message)-written, "\n");
		FPRINTF(stderr, "%s", message);
	}
}

struct starpu_codelet cl_check_scale =
{
	.cpu_funcs = {fmultiple_check_scale},
	.cpu_funcs_name = {"fmultiple_check_scale"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "fmultiple_check_scale"
};

