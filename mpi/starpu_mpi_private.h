/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_MPI_PRIVATE_H__
#define __STARPU_MPI_PRIVATE_H__

#include <starpu.h>
#include <common/config.h>
#include "starpu_mpi.h"
#include "starpu_mpi_fxt.h"
#include <common/list.h>
#include <common/utils.h>
#include <pthread.h>

//#define STARPU_MPI_VERBOSE	1

#ifdef STARPU_MPI_VERBOSE
#  define _STARPU_MPI_DEBUG(fmt, args ...) { if (!getenv("STARPU_SILENT")) { \
    						int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);                     \
                                                int yyy; for(yyy=0 ; yyy<=rank ; yyy++) fprintf(stderr, "    ");    \
                                                fprintf(stderr, "[%d][starpu_mpi][%s] " fmt , rank, __func__ ,##args); \
                                                fflush(stderr); }}
#else
#  define _STARPU_MPI_DEBUG(fmt, args ...)
#endif

#ifdef STARPU_MPI_VERBOSE0
#  define _STARPU_MPI_LOG_IN()             { if (!getenv("STARPU_SILENT")) { \
                                               int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);                        \
                                               int yyy; for(yyy=0 ; yyy<=rank ; yyy++) fprintf(stderr, "    ");      \
                                               fprintf(stderr, "[%d][starpu_mpi][%s] -->\n", rank, __func__ ); \
                                               fflush(stderr); }}
#  define _STARPU_MPI_LOG_OUT()            { if (!getenv("STARPU_SILENT")) { \
                                               int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);                        \
                                               int yyy; for(yyy=0 ; yyy<=rank ; yyy++) fprintf(stderr, "    ");      \
                                               fprintf(stderr, "[%d][starpu_mpi][%s] <--\n", rank, __func__ ); \
                                               fflush(stderr); }}
#else
#  define _STARPU_MPI_LOG_IN()
#  define _STARPU_MPI_LOG_OUT()
#endif

#define SEND_REQ	0
#define RECV_REQ	1
#define WAIT_REQ        2
#define TEST_REQ        3
#define BARRIER_REQ     4

LIST_TYPE(starpu_mpi_req,
	/* description of the data at StarPU level */
	starpu_data_handle_t data_handle;

	/* description of the data to be sent/received */
	MPI_Datatype datatype;

	/* who are we talking to ? */
	int srcdst;
	int mpi_tag;
	MPI_Comm comm;

	void (*func)(struct starpu_mpi_req_s *);

	MPI_Status *status;
	MPI_Request request;
	int *flag;

	int ret;
	pthread_mutex_t req_mutex;
	pthread_cond_t req_cond;

	unsigned request_type; /* 0 send, 1 recv */

	unsigned submitted;
	unsigned completed;

	/* In the case of a Wait/Test request, we are going to post a request
	 * to test the completion of another request */
	struct starpu_mpi_req_s *other_request;

	/* in the case of detached requests */
	unsigned detached;
	void *callback_arg;
	void (*callback)(void *);
);

#endif // __STARPU_MPI_PRIVATE_H__
