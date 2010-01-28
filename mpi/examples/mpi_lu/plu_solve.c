/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include "pxlu.h"

static STARPU_PLU(compute_ax_block)(unsigned size, unsigned nblocks,
				 TYPE *block_data, TYPE *sub_x, TYPE *sub_y)
{
	CPU_GEMV("N", size/nblocks, size/nblocks, 1.0, block_data, size/nblocks, sub_x, 1, 1.0, sub_y, 1);
}

/* y is only valid on node 0 */
void STARPU_PLU(compute_ax)(unsigned size, TYPE *x, TYPE *y, unsigned nblocks, int rank)
{
	/* Create temporary buffers where all MPI processes are going to
	 * compute Ai x = yi where Ai is the matrix containing the blocks of A
	 * affected to process i, and 0 everywhere else. We then have y as the
	 * sum of all yi. */
	TYPE *yi = calloc(size, sizeof(TYPE));

	/* Compute Aix = yi */
	unsigned long i,j;
	for (j = 0; j < nblocks; j++)
	{
		for (i = 0; i < nblocks; i++)
		{
			if (get_block_rank(i, j) == rank)
			{
				/* That block belongs to the current MPI process */
				TYPE *block_data = STARPU_PLU(get_block)(j, i);
				TYPE *sub_x = &x[i*(size/nblocks)];
				TYPE *sub_yi = &yi[j*(size/nblocks)];

				STARPU_PLU(compute_ax_block)(size, nblocks, block_data, sub_x, sub_yi);
			}
		}
	}

	/* Compute the Sum of all yi = y */
	MPI_Reduce(yi, y, size, MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
}
