/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_template.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_package.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_service_comms.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_stats.h>

starpu_pthread_mutex_t           ft_mutex;
int                              _my_rank;

int starpu_mpi_checkpoint_init(void)
{
	STARPU_PTHREAD_MUTEX_INIT(&ft_mutex, NULL);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &_my_rank); //TODO: check compatibility with several Comms behaviour
	starpu_mpi_ft_service_lib_init(_ack_msg_recv_cb, _cp_discard_message_recv_cb);
	checkpoint_template_lib_init();
	_starpu_mpi_checkpoint_tracker_init();
	checkpoint_package_init();
	_STARPU_MPI_FT_STATS_INIT();
	return 0;
}

int starpu_mpi_checkpoint_shutdown(void)
{
	checkpoint_template_lib_quit();
	checkpoint_package_shutdown();
	_starpu_mpi_checkpoint_tracker_shutdown();
	STARPU_PTHREAD_MUTEX_DESTROY(&ft_mutex);
	_STARPU_MPI_FT_STATS_WRITE_TO_FD(stderr);
	_STARPU_MPI_FT_STATS_SHUTDOWN();
	return 0;
}

void starpu_mpi_ft_progress(void)
{
	starpu_mpi_ft_service_progress();
}

int starpu_mpi_ft_busy()
{
	return starpu_mpi_ft_service_lib_busy();
}
