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

#ifndef __PXLU_H__
#define __PXLU_H__

/* for USE_CUDA */
#include <starpu_config.h>
#include <starpu.h>

#include <common/blas.h>

#include <starpu_mpi.h>

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

double STARPU_PLU(plu_main)(unsigned nblocks, int rank, int world_size);

TYPE *STARPU_PLU(reconstruct_matrix)(unsigned size, unsigned nblocks);
void STARPU_PLU(compute_lu_matrix)(unsigned size, unsigned nblocks, TYPE *Asaved);

void STARPU_PLU(compute_ax)(unsigned size, TYPE *x, TYPE *y, unsigned nblocks, int rank);
void STARPU_PLU(compute_lux)(unsigned size, TYPE *x, TYPE *y, unsigned nblocks, int rank);
starpu_data_handle STARPU_PLU(get_block_handle)(unsigned i, unsigned j);
TYPE *STARPU_PLU(get_block)(unsigned i, unsigned j);
starpu_data_handle STARPU_PLU(get_tmp_11_block_handle)(void);
starpu_data_handle STARPU_PLU(get_tmp_12_block_handle)(unsigned j);
starpu_data_handle STARPU_PLU(get_tmp_21_block_handle)(unsigned i);

void STARPU_PLU(display_data_content)(TYPE *data, unsigned blocksize);

#endif // __PXLU_H__
