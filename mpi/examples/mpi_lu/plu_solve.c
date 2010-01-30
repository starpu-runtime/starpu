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

void STARPU_PLU(display_data_content)(TYPE *data, unsigned blocksize)
{
	fprintf(stderr, "DISPLAY BLOCK\n");

	unsigned i, j;
	for (j = 0; j < blocksize; j++)
	{
		for (i = 0; i < blocksize; i++)
		{
			fprintf(stderr, "%f ", data[j+i*blocksize]);
		}
		fprintf(stderr, "\n");
	}

	fprintf(stderr, "****\n");
}

static STARPU_PLU(compute_ax_block)(unsigned block_size, TYPE *block_data, TYPE *sub_x, TYPE *sub_y)
{
	CPU_GEMV("N", block_size, block_size, 1.0, block_data, block_size, sub_x, 1, 1.0, sub_y, 1);
}

void STARPU_PLU(extract_upper)(unsigned block_size, TYPE *inblock, TYPE *outblock)
{
	unsigned li, lj;
	for (lj = 0; lj < block_size; lj++)
	{
		/* Upper block diag is 1 */
		outblock[lj*(block_size + 1)] = (TYPE)1.0;

		for (li = lj + 1; li < block_size; li++)
		{
			outblock[lj + li*block_size] = inblock[lj + li*block_size];
		}
	}
}

static STARPU_PLU(compute_ax_block_upper)(unsigned size, unsigned nblocks,
				 TYPE *block_data, TYPE *sub_x, TYPE *sub_y)
{
	unsigned block_size = size/nblocks;

	fprintf(stderr, "KEEP UPPER\n");
	STARPU_PLU(display_data_content)(block_data, block_size);

	/* Take a copy of the upper part of the diagonal block */
	TYPE *upper_block_copy = calloc((block_size)*(block_size), sizeof(TYPE));
	STARPU_PLU(extract_upper)(block_size, block_data, upper_block_copy);
		
	STARPU_PLU(display_data_content)(upper_block_copy, block_size);

	STARPU_PLU(compute_ax_block)(size/nblocks, upper_block_copy, sub_x, sub_y);
	
	free(upper_block_copy);
}


void STARPU_PLU(extract_lower)(unsigned block_size, TYPE *inblock, TYPE *outblock)
{
	unsigned li, lj;
	for (lj = 0; lj < block_size; lj++)
	{
		for (li = 0; li <= lj; li++)
		{
			outblock[lj + li*block_size] = inblock[lj + li*block_size];
		}
	}
}


TYPE *STARPU_PLU(reconstruct_matrix)(unsigned size, unsigned nblocks)
{
	TYPE *bigmatrix = calloc(size*size, sizeof(TYPE));

	unsigned block_size = size/nblocks;

	unsigned bi, bj;
	for (bj = 0; bj < nblocks; bj++)
	for (bi = 0; bi < nblocks; bi++)
	{
		TYPE *block = STARPU_PLU(get_block)(bj, bi);
		//TYPE *block = STARPU_PLU(get_block)(bj, bi);

		unsigned j, i;
		for (j = 0; j < block_size; j++)
		for (i = 0; i < block_size; i++)
		{
			bigmatrix[(j + bj*block_size)+(i+bi*block_size)*size] =
								block[j+i*block_size];
		}
	}

	return bigmatrix;
}

static TYPE *reconstruct_lower(unsigned size, unsigned nblocks)
{
	TYPE *lower = calloc(size*size, sizeof(TYPE));

	TYPE *bigmatrix = STARPU_PLU(reconstruct_matrix)(size, nblocks);

	STARPU_PLU(extract_lower)(size, bigmatrix, lower); 

	return lower;
}

static TYPE *reconstruct_upper(unsigned size, unsigned nblocks)
{
	TYPE *upper = calloc(size*size, sizeof(TYPE));

	TYPE *bigmatrix = STARPU_PLU(reconstruct_matrix)(size, nblocks);

	STARPU_PLU(extract_upper)(size, bigmatrix, upper); 

	return upper;
}


void STARPU_PLU(compute_lu_matrix)(unsigned size, unsigned nblocks)
{
	fprintf(stderr, "ALL\n\n");
	TYPE *all_r = STARPU_PLU(reconstruct_matrix)(size, nblocks);
	STARPU_PLU(display_data_content)(all_r, size);

	fprintf(stderr, "\nLOWER\n");
	TYPE *lower_r = reconstruct_lower(size, nblocks);
	STARPU_PLU(display_data_content)(lower_r, size);

	fprintf(stderr, "\nUPPER\n");
	TYPE *upper_r = reconstruct_upper(size, nblocks);
	STARPU_PLU(display_data_content)(upper_r, size);

	TYPE *lu_r = calloc(size*size, sizeof(TYPE));
	CPU_TRMM("R", "U", "N", "U", size, size, 1.0f, lower_r, size, upper_r, size);

	fprintf(stderr, "\nLU\n");
	STARPU_PLU(display_data_content)(lower_r, size);
}

static STARPU_PLU(compute_ax_block_lower)(unsigned size, unsigned nblocks,
				 TYPE *block_data, TYPE *sub_x, TYPE *sub_y)
{
	unsigned block_size = size/nblocks;

	fprintf(stderr, "KEEP LOWER\n");
	STARPU_PLU(display_data_content)(block_data, block_size);

	/* Take a copy of the upper part of the diagonal block */
	TYPE *lower_block_copy = calloc((block_size)*(block_size), sizeof(TYPE));
	STARPU_PLU(extract_lower)(block_size, block_data, lower_block_copy);

	STARPU_PLU(display_data_content)(lower_block_copy, block_size);

	STARPU_PLU(compute_ax_block)(size/nblocks, lower_block_copy, sub_x, sub_y);
	
	free(lower_block_copy);
}

void STARPU_PLU(compute_lux)(unsigned size, TYPE *x, TYPE *y, unsigned nblocks, int rank)
{
	/* Create temporary buffers where all MPI processes are going to
	 * compute Ui x = yi where Ai is the matrix containing the blocks of U
	 * affected to process i, and 0 everywhere else. We then have y as the
	 * sum of all yi. */
	TYPE *yi = calloc(size, sizeof(TYPE));

	unsigned block_size = size/nblocks;

	/* Compute UiX = Yi */
	unsigned long i,j;
	for (j = 0; j < nblocks; j++)
	{
		if (get_block_rank(j, j) == rank)
		{
			TYPE *block_data = STARPU_PLU(get_block)(j, j);
			TYPE *sub_x = &x[j*(block_size)];
			TYPE *sub_yi = &yi[j*(block_size)];

			STARPU_PLU(compute_ax_block_upper)(size, nblocks, block_data, sub_x, sub_yi);
		}

		for (i = j + 1; i < nblocks; i++)
		{
			if (get_block_rank(i, j) == rank)
			{
				/* That block belongs to the current MPI process */
				TYPE *block_data = STARPU_PLU(get_block)(j, i);
				TYPE *sub_x = &x[i*(block_size)];
				TYPE *sub_yi = &yi[j*(block_size)];

				STARPU_PLU(compute_ax_block)(size/nblocks, block_data, sub_x, sub_yi);
			}
		}
	}

	/* Grab Sum Yi in X */
	MPI_Reduce(yi, x, size, MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	memset(yi, 0, size*sizeof(TYPE));

	unsigned ind;
	if (rank == 0)
	{
		fprintf(stderr, "INTERMEDIATE\n");
		for (ind = 0; ind < STARPU_MIN(10, size); ind++)
		{
			fprintf(stderr, "x[%d] = %f\n", ind, (float)x[ind]);
		}
		fprintf(stderr, "****\n");
	}

	/* Everyone needs x */
	int bcst_ret;
	bcst_ret = MPI_Bcast(&x, size, MPI_TYPE, 0, MPI_COMM_WORLD);
	STARPU_ASSERT(bcst_ret == MPI_SUCCESS);

	/* Compute LiX = Yi (with X = UX) */
	for (j = 0; j < nblocks; j++)
	{
		if (j > 0)
		for (i = 0; i < j; i++)
		{
			if (get_block_rank(i, j) == rank)
			{
				/* That block belongs to the current MPI process */
				TYPE *block_data = STARPU_PLU(get_block)(j, i);
				TYPE *sub_x = &x[i*(block_size)];
				TYPE *sub_yi = &yi[j*(block_size)];

				STARPU_PLU(compute_ax_block)(size/nblocks, block_data, sub_x, sub_yi);
			}
		}

		if (get_block_rank(j, j) == rank)
		{
			TYPE *block_data = STARPU_PLU(get_block)(j, j);
			TYPE *sub_x = &x[j*(block_size)];
			TYPE *sub_yi = &yi[j*(block_size)];

			STARPU_PLU(compute_ax_block_lower)(size, nblocks, block_data, sub_x, sub_yi);
		}
	}

	/* Grab Sum Yi in Y */
	MPI_Reduce(yi, y, size, MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

	free(yi);
}


/* x and y must be valid (at least) on 0 */
void STARPU_PLU(compute_ax)(unsigned size, TYPE *x, TYPE *y, unsigned nblocks, int rank)
{
	/* Send x to everyone */
	int bcst_ret;
	bcst_ret = MPI_Bcast(&x, size, MPI_TYPE, 0, MPI_COMM_WORLD);
	STARPU_ASSERT(bcst_ret == MPI_SUCCESS);

	if (rank == 0)
	{
		unsigned ind;
		for (ind = 0; ind < STARPU_MIN(10, size); ind++)
			fprintf(stderr, "x[%d] = %f\n", ind, (float)x[ind]);

		fprintf(stderr, "Compute AX = B\n");
	}

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

				STARPU_PLU(compute_ax_block)(size/nblocks, block_data, sub_x, sub_yi);
			}
		}
	}

	/* Compute the Sum of all yi = y */
	MPI_Reduce(yi, y, size, MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

	free(yi);
}
