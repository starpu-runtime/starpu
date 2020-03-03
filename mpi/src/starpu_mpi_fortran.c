/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi.h>
#include <common/config.h>
#include "starpu_mpi_private.h"

#ifdef HAVE_MPI_COMM_F2C
/* Fortran related functions */
struct _starpu_mpi_argc_argv *fstarpu_mpi_argcv_alloc(int argc, int initialize_mpi, int comm_present, MPI_Fint comm)
{
	struct _starpu_mpi_argc_argv *argcv;
	_STARPU_MPI_CALLOC(argcv, 1,sizeof(*argcv));
	argcv->initialize_mpi = initialize_mpi;
	if (comm_present)
	{
		argcv->comm = MPI_Comm_f2c(comm);
	}
	else
	{
		argcv->comm = MPI_COMM_WORLD;
	}
	argcv->fargc = argc;
	argcv->argc = &argcv->fargc;
	_STARPU_MPI_CALLOC(argcv->fargv, argc, sizeof(char *));
	argcv->argv = &argcv->fargv;
	return argcv;
}

void fstarpu_mpi_argcv_set_arg(struct _starpu_mpi_argc_argv *argcv, int i, int len, char *_s)
{
	STARPU_ASSERT(len >= 0);
	STARPU_ASSERT(i >= 0 && i < argcv->fargc);
	char *s;
	_STARPU_MPI_MALLOC(s, len+1);
	memcpy(s, _s, len);
	s[len] = '\0';
	argcv->fargv[i] = s;
}

void fstarpu_mpi_argcv_free(struct _starpu_mpi_argc_argv *argcv)
{
	if (argcv->fargv != NULL)
	{
		int i;
		for (i=0; i<argcv->fargc; i++)
		{
			free(argcv->fargv[i]);
		}
		free(argcv->fargv);
	}
	free(argcv);
}

starpu_mpi_req *fstarpu_mpi_req_alloc(void)
{
	starpu_mpi_req *req;
	_STARPU_MPI_CALLOC(req, 1, sizeof(starpu_mpi_req));
	return req;
}

void fstarpu_mpi_req_free(starpu_mpi_req *req)
{
	free(req);
}

MPI_Status *fstarpu_mpi_status_alloc(void)
{
	MPI_Status *s;
	_STARPU_MPI_CALLOC(s, 1, sizeof(MPI_Status));
	return s;
}

void fstarpu_mpi_status_free(MPI_Status *status)
{
	free(status);
}

int fstarpu_mpi_barrier(MPI_Fint comm)
{
	return starpu_mpi_barrier(MPI_Comm_f2c(comm));
}

int fstarpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int src, int mpi_tag, MPI_Fint comm, void (*callback)(void *), void *arg, int seq_const)
{
	return starpu_mpi_irecv_detached_sequential_consistency(data_handle, src, mpi_tag, MPI_Comm_f2c(comm), callback, arg, seq_const);
}

int fstarpu_mpi_init_c(struct _starpu_mpi_argc_argv *argcv)
{
	return starpu_mpi_init_comm(argcv->argc, argcv->argv, argcv->initialize_mpi, argcv->comm);
}

void fstarpu_mpi_get_data_on_node(MPI_Fint comm, starpu_data_handle_t data_handle, int node)
{
	starpu_mpi_get_data_on_node(MPI_Comm_f2c(comm), data_handle, node);
}

void fstarpu_mpi_get_data_on_node_detached(MPI_Fint comm, starpu_data_handle_t data_handle, int node, void (*callback)(void *), void *arg)
{
	starpu_mpi_get_data_on_node_detached(MPI_Comm_f2c(comm), data_handle, node, callback, arg);
}

void fstarpu_mpi_redux_data(MPI_Fint comm, starpu_data_handle_t data_handle)
{
	starpu_mpi_redux_data(MPI_Comm_f2c(comm), data_handle);
}

/* scatter/gather */
int fstarpu_mpi_scatter_detached(starpu_data_handle_t *data_handles, int cnt, int root, MPI_Fint comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg)
{
	return starpu_mpi_scatter_detached(data_handles, cnt, root, MPI_Comm_f2c(comm), scallback, sarg, rcallback, rarg);
}

int fstarpu_mpi_gather_detached(starpu_data_handle_t *data_handles, int cnt, int root, MPI_Fint comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg)
{
	return starpu_mpi_gather_detached(data_handles, cnt, root, MPI_Comm_f2c(comm), scallback, sarg, rcallback, rarg);
}

/* isend/irecv detached unlock tag */
int fstarpu_mpi_isend_detached_unlock_tag(starpu_data_handle_t data_handle, int dst, int mpi_tag, MPI_Fint comm, starpu_tag_t *starpu_tag)
{
	return starpu_mpi_isend_detached_unlock_tag(data_handle, dst, mpi_tag, MPI_Comm_f2c(comm), *starpu_tag);
}

int fstarpu_mpi_irecv_detached_unlock_tag(starpu_data_handle_t data_handle, int src, int mpi_tag, MPI_Fint comm, starpu_tag_t *starpu_tag)
{
	return starpu_mpi_irecv_detached_unlock_tag(data_handle, src, mpi_tag, MPI_Comm_f2c(comm), *starpu_tag);
}

/* isend/irecv array detached unlock tag */
int fstarpu_mpi_isend_array_detached_unlock_tag(int array_size, starpu_data_handle_t *data_handles, int *dsts, int *mpi_tags, MPI_Fint *_comms, starpu_tag_t *starpu_tag)
{
	MPI_Comm comms[array_size];
	int i;
	for (i = 0; i < array_size; i++)
	{
		comms[i] = MPI_Comm_f2c(_comms[i]);
	}
	int ret = starpu_mpi_isend_array_detached_unlock_tag((unsigned)array_size, data_handles, dsts, mpi_tags, comms, *starpu_tag);
	return ret;
}

int fstarpu_mpi_irecv_array_detached_unlock_tag(int array_size, starpu_data_handle_t *data_handles, int *srcs, int *mpi_tags, MPI_Fint *_comms, starpu_tag_t *starpu_tag)
{
	MPI_Comm comms[array_size];
	int i;
	for (i = 0; i < array_size; i++)
	{
		comms[i] = MPI_Comm_f2c(_comms[i]);
	}
	int ret = starpu_mpi_irecv_array_detached_unlock_tag((unsigned)array_size, data_handles, srcs, mpi_tags, comms, *starpu_tag);
	return ret;
}

/* isend/irecv */
int fstarpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dst, int mpi_tag, MPI_Fint comm)
{
	return starpu_mpi_isend(data_handle, req, dst, mpi_tag, MPI_Comm_f2c(comm));
}

int fstarpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *req, int src, int mpi_tag, MPI_Fint comm)
{
	return starpu_mpi_irecv(data_handle, req, src, mpi_tag, MPI_Comm_f2c(comm));
}

/* send/recv */
int fstarpu_mpi_send(starpu_data_handle_t data_handle, int dst, int mpi_tag, MPI_Fint comm)
{
	return starpu_mpi_send(data_handle, dst, mpi_tag, MPI_Comm_f2c(comm));
}

int fstarpu_mpi_recv(starpu_data_handle_t data_handle, int src, int mpi_tag, MPI_Fint comm, MPI_Status *status)
{
	return starpu_mpi_recv(data_handle, src, mpi_tag, MPI_Comm_f2c(comm), status);
}

/* isend/irecv detached */
int fstarpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dst, int mpi_tag, MPI_Fint comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_isend_detached(data_handle, dst, mpi_tag, MPI_Comm_f2c(comm), callback, arg);
}

int fstarpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int src, int mpi_tag, MPI_Fint comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_irecv_detached(data_handle, src, mpi_tag, MPI_Comm_f2c(comm), callback, arg);
}

/* issend / issend detached */
int fstarpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dst, int mpi_tag, MPI_Fint comm)
{
	return starpu_mpi_issend(data_handle, req, dst, mpi_tag, MPI_Comm_f2c(comm));
}

int fstarpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dst, int mpi_tag, MPI_Fint comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_issend_detached(data_handle, dst, mpi_tag, MPI_Comm_f2c(comm), callback, arg);
}

/* cache */
void fstarpu_mpi_cache_flush(MPI_Fint comm, starpu_data_handle_t data_handle)
{
	return starpu_mpi_cache_flush(MPI_Comm_f2c(comm), data_handle);
}

void fstarpu_mpi_cache_flush_all_data(MPI_Fint comm)
{
	return starpu_mpi_cache_flush_all_data(MPI_Comm_f2c(comm));
}

int fstarpu_mpi_comm_size(MPI_Fint comm, int *size)
{
	return starpu_mpi_comm_size(MPI_Comm_f2c(comm), size);
}

int fstarpu_mpi_comm_rank(MPI_Fint comm, int *rank)
{
	return starpu_mpi_comm_rank(MPI_Comm_f2c(comm), rank);
}

MPI_Fint fstarpu_mpi_world_comm()
{
	return MPI_Comm_c2f(MPI_COMM_WORLD);
}

void fstarpu_mpi_data_register_comm(starpu_data_handle_t handle, int tag, int rank, MPI_Fint comm)
{
	return starpu_mpi_data_register_comm(handle, tag, rank, MPI_Comm_f2c(comm));
}

void fstarpu_mpi_data_register(starpu_data_handle_t handle, int tag, int rank)
{
	return starpu_mpi_data_register_comm(handle, tag, rank, MPI_COMM_WORLD);
}

void fstarpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Fint comm)
{
	return starpu_mpi_data_set_rank_comm(handle, rank, MPI_Comm_f2c(comm));
}

void fstarpu_mpi_data_set_rank(starpu_data_handle_t handle, int rank)
{
	return starpu_mpi_data_set_rank_comm(handle, rank, MPI_COMM_WORLD);
}

void fstarpu_mpi_data_migrate(MPI_Fint comm, starpu_data_handle_t handle, int rank)
{
	return starpu_mpi_data_migrate(MPI_Comm_f2c(comm), handle, rank);
}

int fstarpu_mpi_wait_for_all(MPI_Fint comm)
{
	return starpu_mpi_wait_for_all(MPI_Comm_f2c(comm));
}
#endif
