/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int _starpu_debug_rank=-1;
int _starpu_debug_level_min=0;
int _starpu_debug_level_max=0;
int _starpu_mpi_tag = 42;
int _starpu_mpi_comm_debug;

int _starpu_mpi_nobind = -1;
int _starpu_mpi_thread_cpuid = -1;
int _starpu_mpi_use_prio = 1;
int _starpu_mpi_fake_world_size = -1;
int _starpu_mpi_fake_world_rank = -1;
int _starpu_mpi_use_coop_sends = 1;

void _starpu_mpi_set_debug_level_min(int level)
{
	_starpu_debug_level_min = level;
}

void _starpu_mpi_set_debug_level_max(int level)
{
	_starpu_debug_level_max = level;
}

int starpu_mpi_get_communication_tag(void)
{
	return _starpu_mpi_tag;
}

void starpu_mpi_set_communication_tag(int tag)
{
	_starpu_mpi_tag = tag;
}

char *_starpu_mpi_get_mpi_error_code(int code)
{
	static char str[MPI_MAX_OBJECT_NAME];
	int len;
	MPI_Error_string(code, str, &len);
	return str;
}

void _starpu_mpi_env_init(void)
{
        _starpu_mpi_comm_debug = starpu_getenv("STARPU_MPI_COMM") != NULL;
	_starpu_mpi_fake_world_size = starpu_get_env_number("STARPU_MPI_FAKE_SIZE");
	_starpu_mpi_fake_world_rank = starpu_get_env_number("STARPU_MPI_FAKE_RANK");
	_starpu_mpi_nobind = starpu_get_env_number_default("STARPU_MPI_NOBIND", 0);
	_starpu_mpi_thread_cpuid = starpu_get_env_number_default("STARPU_MPI_THREAD_CPUID", -1);
	_starpu_mpi_use_prio = starpu_get_env_number_default("STARPU_MPI_PRIORITIES", 1);
	_starpu_mpi_use_coop_sends = starpu_get_env_number_default("STARPU_MPI_COOP_SENDS", 1);
	_starpu_debug_level_min = starpu_get_env_number_default("STARPU_MPI_DEBUG_LEVEL_MIN", 0);
	_starpu_debug_level_max = starpu_get_env_number_default("STARPU_MPI_DEBUG_LEVEL_MAX", 0);
}
