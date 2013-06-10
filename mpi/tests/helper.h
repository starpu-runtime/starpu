/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <errno.h>

#define STARPU_TEST_SKIPPED 77

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define FPRINTF_MPI(fmt, ...) do { if (!getenv("STARPU_SILENT")) { \
    						int _disp_rank; MPI_Comm_rank(MPI_COMM_WORLD, &_disp_rank);       \
                                                fprintf(stderr, "[%d][starpu_mpi][%s] " fmt , _disp_rank, __starpu_func__ ,## __VA_ARGS__); \
                                                fflush(stderr); }} while(0);

