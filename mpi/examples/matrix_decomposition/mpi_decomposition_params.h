/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __MPI_CHOLESKY_PARAMS_H__
#define __MPI_CHOLESKY_PARAMS_H__

#define BLOCKSIZE       (size/nblocks)

extern unsigned size;
extern unsigned nblocks;
extern unsigned nbigblocks;
extern unsigned noprio;
extern unsigned check;
extern unsigned display;
extern int dblockx;
extern int dblocky;
extern unsigned long checkpoint_period;
extern int checkpoint_enabled;

enum submission
{
	TRIANGLES,
	COLUMNS,
	ANTIDIAGONALS,
	PRIOS,
};
extern enum submission submission;

void parse_args(int argc, char **argv, int nodes);

#endif // __MPI_CHOLESKY_PARAMS_H__

