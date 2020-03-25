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

#include <errno.h>
#include <starpu_mpi.h>

#ifdef STARPU_HAVE_VALGRIND_H
#include <valgrind/valgrind.h>
#endif

#ifdef STARPU_HAVE_HELGRIND_H
#include <valgrind/helgrind.h>
#endif

#define STARPU_TEST_SKIPPED 77

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define FPRINTF_MPI(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) { \
    						int _disp_rank; starpu_mpi_comm_rank(MPI_COMM_WORLD, &_disp_rank);       \
                                                fprintf(ofile, "[%d][starpu_mpi][%s] " fmt , _disp_rank, __starpu_func__ ,## __VA_ARGS__); \
                                                fflush(ofile); }} while(0);

