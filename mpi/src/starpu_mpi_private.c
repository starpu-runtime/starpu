/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <starpu_mpi_private.h>

int _debug_rank=-1;
int _debug_level=3;
int starpu_mpi_tag = 42;

void _starpu_mpi_set_debug_level(int level)
{
	_debug_level = level;
}

int starpu_mpi_get_starpu_mpi_tag(void)
{
	return starpu_mpi_tag;
}

void starpu_mpi_set_starpu_mpi_tag(int tag)
{
	starpu_mpi_tag = tag;
}
