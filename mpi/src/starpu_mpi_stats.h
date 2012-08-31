/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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

#include <stdlib.h>
#include <mpi.h>

void _starpu_mpi_comm_amounts_init(MPI_Comm comm);
void _starpu_mpi_comm_amounts_free();
void _starpu_mpi_comm_amounts_inc(MPI_Comm comm, unsigned dst, MPI_Datatype datatype, int count);
void _starpu_mpi_comm_amounts_display(int node);

