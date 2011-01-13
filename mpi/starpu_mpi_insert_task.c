/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#include <stdarg.h>
#include <mpi.h>

#include <starpu.h>
#include <starpu_data.h>
#include <common/utils.h>
#include <util/starpu_insert_task_utils.h>

#define STARPU_MPI_VERBOSE	1
#include <starpu_mpi_private.h>

/* Whether we are allowed to keep copies of remote data. Does not work
 * yet: the sender has to know whether the receiver has it, keeping it
 * in an array indexed by node numbers. */
//#define MPI_CACHE

int starpu_mpi_insert_task(MPI_Comm comm, starpu_codelet *codelet, ...) {
        int arg_type;
        va_list varg_list;
        int me, do_execute;
	size_t arg_buffer_size = 0;
        int nb_buffers;
        int dest;

	MPI_Comm_rank(comm, &me);

        /* Get the number of buffers and the size of the arguments */
	va_start(varg_list, codelet);
        starpu_insert_task_get_sizes(&arg_buffer_size, &nb_buffers, varg_list);

	/* Find out whether we are to execute the data because we own the data to be written to. */
        do_execute = -1;
	va_start(varg_list, codelet);
	while ((arg_type = va_arg(varg_list, int)) != 0) {
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH) {
                        starpu_data_handle data = va_arg(varg_list, starpu_data_handle);
                        if (arg_type & STARPU_W) {
                                if (!data) {
                                        /* We don't have anything allocated for this.
                                         * The application knows we won't do anything
                                         * about this task */
                                        /* Yes, the app could actually not call
                                         * insert_task at all itself, this is just a
                                         * safeguard. */
                                        _STARPU_MPI_DEBUG("oh oh\n");
                                        return;
                                }
                                int mpi_rank = starpu_data_get_rank(data);
                                if (mpi_rank == me) {
                                        if (do_execute == 0) {
                                                _STARPU_ERROR("erh? incoherent!\n");
                                        }
                                        else {
                                                do_execute = 1;
                                        }
                                }
                                else {
                                        if (do_execute == 1) {
                                                _STARPU_ERROR("erh? incoherent!\n");
                                        }
                                        else {
                                                do_execute = 0;
                                                dest = mpi_rank;
                                                /* That's the rank which needs the data to be sent to */
                                        }
                                }
                        }
                }
	}
	va_end(varg_list);
        assert(do_execute != -1);

        starpu_mpi_req *req = malloc(nb_buffers * sizeof(starpu_mpi_req));
        int nb_req=0;

        /* Send and receive data as requested */
	va_start(varg_list, codelet);
	while ((arg_type = va_arg(varg_list, int)) != 0) {
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH) {
                        starpu_data_handle data = va_arg(varg_list, starpu_data_handle);
                        if (arg_type & STARPU_R) {
                                int mpi_rank = starpu_data_get_rank(data);
                                /* The task needs to read this data */
                                if (do_execute && mpi_rank != me) {
                                        _STARPU_MPI_DEBUG("Receive data from %d\n", mpi_rank);
                                        /* I will have to execute but I don't have the data, receive */
#ifdef MPI_CACHE
                                        if (!starpu_allocated(data))
#endif
                                                {
                                                        starpu_mpi_irecv(data, &req[nb_req], mpi_rank, 0, comm);
                                                        nb_req++;
                                                }
                                }
                                if (!do_execute && mpi_rank == me) {
                                        /* Somebody else will execute it, and I have the data, send it. */
                                        /* FIXME CACHE: we need to know whether the receiver has it. */
                                        _STARPU_MPI_DEBUG("Send data to %d\n", dest);
                                        starpu_mpi_isend(data, &req[nb_req], dest, 0, comm);
                                        nb_req++;
                                }
                        }
                }
        }
	va_end(varg_list);

        /* If some MPI communications have been posted, wait until they are finished */
        _STARPU_MPI_DEBUG("Waiting for %d request(s)\n", nb_req);
        int nb_waiting_requests=nb_req;
        while(nb_waiting_requests) {
                //_STARPU_MPI_DEBUG("Testing %d request(s)\n", nb_waiting_requests);
                int r=0;
                for(r=0 ; r<nb_req ; r++) {
                        if (req[r]) {
                                int finished = 0;
                                MPI_Status status;
                                //_STARPU_MPI_DEBUG("Testing request %d\n", r);
                                starpu_mpi_test(&req[r], &finished, &status);
                                STARPU_ASSERT(finished != -1);
                                if(finished) {
                                        req[r] = NULL;
                                        nb_waiting_requests--;
                                }
                        }
                }
        }
        _STARPU_MPI_DEBUG("All requests processed\n", nb_req);
        free(req);

	if (do_execute) {
                _STARPU_MPI_DEBUG("Execution of the codelet\n");
                va_start(varg_list, codelet);
                struct starpu_task *task = starpu_task_create();
                task->synchronous = 1;
                int ret = starpu_insert_task_create_and_submit(arg_buffer_size, codelet, &task, varg_list);
                _STARPU_MPI_DEBUG("ret: %d\n", ret);
                STARPU_ASSERT(ret==0);
                //                ret = starpu_task_wait(task);
                //                _STARPU_MPI_DEBUG("ret: %d\n", ret);
                //                STARPU_ASSERT(ret==0);
        }

	/* No need to handle W, as we assume (and check) that task
	 * write in data that they own */

	va_start(varg_list, codelet);
	while ((arg_type = va_arg(varg_list, int)) != 0) {
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH) {
                        starpu_data_handle data = va_arg(varg_list, starpu_data_handle);
#ifdef MPI_CACHE
                        if (arg_type & STARPU_W) {
                                if (do_execute) {
                                        /* FIXME: I need to note that all
                                         * copies I've sent to neighbours are
                                         * now invalid */
                                }
                                else {
                                        /* Somebody else will write to the data, so discard our cached copy if any */
                                        /* TODO: starpu_mpi could just remember itself. */
                                        if (starpu_allocated(data))
                                                starpu_deallocate(data);
                                }
                        }
#else
                        /* We allocated a temporary buffer for the received data, now drop it */
                        if ((arg_type & STARPU_R) && do_execute) {
                                int mpi_rank = starpu_data_get_rank(data);
                                if (mpi_rank != me) {
                                        //                                        starpu_deallocate(data);
                                }
                        }
#endif
                }
        }
	va_end(varg_list);
}
