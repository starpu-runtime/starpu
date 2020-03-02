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
#include <starpu_config.h>
#include "../../tests/helper.h"

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define FPRINTF_MPI(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) { \
			int _disp_rank; starpu_mpi_comm_rank(MPI_COMM_WORLD, &_disp_rank); \
			fprintf(ofile, "[%d][starpu_mpi][%s] " fmt , _disp_rank, __starpu_func__ ,## __VA_ARGS__); \
			fflush(ofile); }} while(0)

#define MPI_INIT_THREAD_real(argc, argv, required) do {	\
		int thread_support;				\
		if (MPI_Init_thread(argc, argv, required, &thread_support) != MPI_SUCCESS) \
		{						\
			fprintf(stderr,"MPI_Init_thread failed\n");	\
			exit(1);					\
		}							\
		if (thread_support == MPI_THREAD_FUNNELED)		\
			fprintf(stderr,"Warning: MPI only has funneled thread support, not serialized, hoping this will work\n"); \
		if (thread_support < MPI_THREAD_FUNNELED)		\
			fprintf(stderr,"Warning: MPI does not have thread support!\n"); } while(0)

#ifdef STARPU_SIMGRID
#define MPI_INIT_THREAD(argc, argv, required, init) do { *(init) = 1 ; } while(0)
#else
#define MPI_INIT_THREAD(argc, argv, required, init) do {	\
		*(init) = 0;                                    \
		MPI_INIT_THREAD_real(argc, argv, required); } while(0)
#endif

