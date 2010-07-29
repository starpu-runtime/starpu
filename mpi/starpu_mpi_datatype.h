/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

#ifndef __STARPU_MPI_DATATYPE_H__
#define __STARPU_MPI_DATATYPE_H__

#include <starpu_mpi.h>

int starpu_mpi_handle_to_datatype(starpu_data_handle data_handle, MPI_Datatype *datatype);
void *starpu_mpi_handle_to_ptr(starpu_data_handle data_handle);

#endif // __STARPU_MPI_DATATYPE_H__
