/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __MPI_TESTS_BURST_HELPER__
#define __MPI_TESTS_BURST_HELPER__

extern int burst_nb_requests;

void burst_init_data(int rank);
void burst_free_data(int rank);
void burst_bidir(int rank);
void burst_unidir(int sender, int receiver, int rank);
void burst_bidir_half_postponed(int rank);
void burst_all(int rank);

#endif /* __MPI_TESTS_BURST_HELPER__ */
