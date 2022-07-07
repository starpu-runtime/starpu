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

#ifndef FT_STARPU_STARPU_MPI_FT_SERVICE_COMMS_H
#define FT_STARPU_STARPU_MPI_FT_SERVICE_COMMS_H

#ifdef __cplusplus
extern "C"
{
#endif

int _starpu_mpi_ft_service_post_special_recv(int tag);
int _starpu_mpi_ft_service_post_send(void* msg, int count, int rank, int tag, MPI_Comm comm, void (*callback)(void *), void* arg);

void starpu_mpi_test_ft_detached_service_requests(void);
int starpu_mpi_ft_service_progress();
int starpu_mpi_ft_service_lib_init(void(*_ack_msg_recv_cb)(void*), void(*cp_info_recv_cb)(void*));
int starpu_mpi_ft_service_lib_busy();

#ifdef __cplusplus
}
#endif

#endif //FT_STARPU_STARPU_MPI_FT_SERVICE_COMMS_H
